//! Minimal GGUF reader/writer for ternary tensor types.
//!
//! Supports reading GGUF files with memory-mapped tensor data,
//! and writing ternary tensors back to GGUF format.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::path::Path;

use crate::planar::PlanarWeights;

/// Custom GGUF quant type for ternary 1.58-bit weights.
pub const GGUF_TYPE_Q1_58: u32 = 100;

/// GGUF magic number: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

/// GGUF file version we support.
const GGUF_VERSION: u32 = 3;

/// GGUF data types (standard + custom).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
    Q1_58 = 100, // Custom ternary type
}

impl GgufType {
    pub fn to_u32(self) -> u32 {
        self as u32
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Descriptor for a tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

/// A read-only GGUF file backed by memory-mapping.
#[derive(Debug)]
pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorDescriptor>,
    _mmap: Mmap,
    data_offset: usize,
}

impl GgufFile {
    /// Open and parse a GGUF file, memory-mapping the tensor data.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path.as_ref())?;

        // Read header
        let magic = read_u32(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: {:#010x}", magic),
            ));
        }

        let version = read_u32(&mut file)?;
        if !(2..=3).contains(&version) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {}", version),
            ));
        }

        let n_tensors_u64 = read_u64(&mut file)?;
        let n_metadata_u64 = read_u64(&mut file)?;

        const MAX_ENTRIES: u64 = 100_000;
        if n_tensors_u64 > MAX_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "n_tensors {} exceeds maximum allowed {}",
                    n_tensors_u64, MAX_ENTRIES
                ),
            ));
        }
        if n_metadata_u64 > MAX_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "n_metadata {} exceeds maximum allowed {}",
                    n_metadata_u64, MAX_ENTRIES
                ),
            ));
        }
        let n_tensors = usize::try_from(n_tensors_u64).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "n_tensors does not fit usize")
        })?;
        let n_metadata = usize::try_from(n_metadata_u64).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "n_metadata does not fit usize")
        })?;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_gguf_string(&mut file)?;
            let val = read_gguf_value(&mut file)?;
            metadata.insert(key, val);
        }

        // Parse tensor descriptors
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_gguf_string(&mut file)?;
            let n_dims = read_u32(&mut file)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut file)?);
            }
            let dtype = read_u32(&mut file)?;
            let offset = read_u64(&mut file)?;
            tensors.push(TensorDescriptor {
                name,
                n_dims,
                dims,
                dtype,
                offset,
            });
        }

        // Data section starts at next 32-byte aligned position
        let current_pos = file.stream_position()? as usize;
        let data_offset = (current_pos + 31) & !31;

        // Memory-map the entire file
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            metadata,
            tensors,
            _mmap: mmap,
            data_offset,
        })
    }

    /// Get raw tensor data bytes for a given tensor descriptor.
    ///
    /// For Q1_58 tensors, returns packed bytes + f32 scales concatenated.
    pub fn tensor_data(&self, tensor: &TensorDescriptor) -> io::Result<&[u8]> {
        let tensor_offset = usize::try_from(tensor.offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor offset does not fit usize",
            )
        })?;
        let start = self
            .data_offset
            .checked_add(tensor_offset)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor offset overflow"))?;
        let byte_size = self.tensor_byte_size(tensor)?;
        let end = start
            .checked_add(byte_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor size overflow"))?;
        if end > self._mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor '{}' data range [{}..{}) exceeds file size {}",
                    tensor.name,
                    start,
                    end,
                    self._mmap.len()
                ),
            ));
        }
        Ok(&self._mmap[start..end])
    }

    fn tensor_byte_size(&self, tensor: &TensorDescriptor) -> io::Result<usize> {
        let n_elements_u64 = tensor
            .dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "tensor dims overflow"))?;
        let n_elements = usize::try_from(n_elements_u64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor too large"))?;
        match tensor.dtype {
            GGUF_TYPE_Q1_58 => {
                // packed bytes (2 bits per element) + f32 scales
                // For a 2D [rows, cols] tensor with group_size from metadata:
                let rows = usize::try_from(tensor.dims.first().copied().unwrap_or(0))
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "rows too large"))?;
                let cols = usize::try_from(tensor.dims.get(1).copied().unwrap_or(0))
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "cols too large"))?;
                let gs = match self
                    .metadata
                    .get("nanochat.group_size")
                    .or_else(|| self.metadata.get("model.group_size"))
                {
                    Some(GgufValue::U32(v)) => *v as usize,
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "missing group_size metadata for Q1_58 tensor '{}'",
                                tensor.name
                            ),
                        ))
                    }
                };
                if gs == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "group_size metadata must be > 0",
                    ));
                }
                if !cols.is_multiple_of(4) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Q1_58 tensor cols {} must be divisible by 4", cols),
                    ));
                }
                if !cols.is_multiple_of(gs) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Q1_58 tensor cols {} must be divisible by group_size {}",
                            cols, gs
                        ),
                    ));
                }
                let kp = cols / 4;
                let gprow = cols / gs;
                let packed = rows.checked_mul(kp).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "packed size overflow")
                })?;
                let scales = rows
                    .checked_mul(gprow)
                    .and_then(|v| v.checked_mul(4))
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "scale size overflow")
                    })?;
                packed.checked_add(scales).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "tensor size overflow")
                })
            }
            0 => n_elements
                .checked_mul(4)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "F32 tensor too large")),
            1 => n_elements
                .checked_mul(2)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "F16 tensor too large")),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported GGUF dtype: {}", tensor.dtype),
            )),
        }
    }

    /// Load a ternary tensor and convert to PlanarWeights.
    pub fn load_planar_weights(&self, name: &str, group_size: usize) -> io::Result<PlanarWeights> {
        let tensor = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("tensor '{}' not found", name),
                )
            })?;

        if tensor.n_dims != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected 2D tensor, got {}D for '{}'", tensor.n_dims, name),
            ));
        }

        let rows = usize::try_from(tensor.dims[0])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "rows too large"))?;
        let cols = usize::try_from(tensor.dims[1])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "cols too large"))?;
        let packed_data = self.tensor_data(tensor)?;
        let kp = cols / 4;
        let gprow = cols / group_size;

        // Extract packed weights and scales from the data
        // Layout: packed bytes followed by scales
        let weight_bytes = rows
            .checked_mul(kp)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "weight size overflow"))?;
        let scale_bytes = rows
            .checked_mul(gprow)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "scale size overflow"))?;

        if packed_data.len() < weight_bytes + scale_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor data too short: {} < {}",
                    packed_data.len(),
                    weight_bytes + scale_bytes
                ),
            ));
        }

        let packed = &packed_data[..weight_bytes];
        let scale_raw = &packed_data[weight_bytes..weight_bytes + scale_bytes];
        let scales: Vec<f32> = scale_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(PlanarWeights::from_packed(
            packed, &scales, rows, cols, group_size,
        ))
    }
}

/// Writer for creating GGUF files with ternary tensors.
pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>, // name, dims, dtype, data
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn add_metadata(&mut self, key: &str, value: GgufValue) {
        self.metadata.push((key.to_string(), value));
    }

    /// Add a ternary tensor (packed weights + scales).
    pub fn add_ternary_tensor(&mut self, name: &str, pw: &PlanarWeights) {
        let kp = pw.cols / 4;
        let gprow = pw.cols / pw.group_size;

        let mut data = Vec::with_capacity(pw.rows * kp + pw.rows * gprow * 4);

        // Packed bytes (row-major)
        data.extend_from_slice(&pw.data[..pw.rows * kp]);

        // Scales (row-major, f32 LE)
        for &s in &pw.scales_rm[..pw.rows * gprow] {
            data.extend_from_slice(&s.to_le_bytes());
        }

        self.tensors.push((
            name.to_string(),
            vec![pw.rows as u64, pw.cols as u64],
            GGUF_TYPE_Q1_58,
            data,
        ));
    }

    /// Add an f32 tensor.
    pub fn add_f32_tensor(&mut self, name: &str, dims: &[u64], data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), dims.to_vec(), 0, bytes)); // type 0 = F32
    }

    /// Write the GGUF file.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes())?;
        file.write_all(&GGUF_VERSION.to_le_bytes())?;
        file.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        file.write_all(&(self.metadata.len() as u64).to_le_bytes())?;

        // Metadata
        for (key, value) in &self.metadata {
            write_gguf_string(&mut file, key)?;
            write_gguf_value(&mut file, value)?;
        }

        // Tensor descriptors
        let mut offset: u64 = 0;
        for (name, dims, dtype, data) in &self.tensors {
            write_gguf_string(&mut file, name)?;
            file.write_all(&(dims.len() as u32).to_le_bytes())?;
            for &d in dims {
                file.write_all(&d.to_le_bytes())?;
            }
            file.write_all(&dtype.to_le_bytes())?;
            file.write_all(&offset.to_le_bytes())?;
            offset += data.len() as u64;
            // Align to 32 bytes
            offset = (offset + 31) & !31;
        }

        // Pad to 32-byte alignment for data section
        let pos = file.stream_position()? as usize;
        let aligned = (pos + 31) & !31;
        let padding = aligned - pos;
        file.write_all(&vec![0u8; padding])?;

        // Tensor data
        for (_, _, _, data) in &self.tensors {
            file.write_all(data)?;
            // Pad to 32-byte alignment
            let pad = ((data.len() + 31) & !31) - data.len();
            if pad > 0 {
                file.write_all(&vec![0u8; pad])?;
            }
        }

        Ok(())
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified GGUF tensor info for writing.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgufType,
    pub offset: u64, // Will be computed by writer
}

/// Simplified GGUF metadata container.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

/// Simplified GGUF file writer (one-shot API).
pub struct GgufFileWriter;

impl GgufFileWriter {
    /// Write a complete GGUF file with metadata and tensors.
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `meta` - Metadata and tensor descriptors
    /// * `tensor_data` - Tensor data arrays (in same order as meta.tensors)
    pub fn write<P: AsRef<Path>>(
        path: P,
        meta: &GgufMetadata,
        tensor_data: &[&[u8]],
    ) -> io::Result<()> {
        if meta.tensors.len() != tensor_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Tensor count mismatch: {} descriptors vs {} data arrays",
                    meta.tensors.len(),
                    tensor_data.len()
                ),
            ));
        }

        let mut file = File::create(path)?;

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes())?;
        file.write_all(&GGUF_VERSION.to_le_bytes())?;
        file.write_all(&(meta.tensors.len() as u64).to_le_bytes())?;
        file.write_all(&(meta.metadata.len() as u64).to_le_bytes())?;

        // Metadata — sort keys for deterministic output
        let mut sorted_keys: Vec<_> = meta.metadata.keys().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            let value = &meta.metadata[key];
            write_gguf_string(&mut file, key)?;
            write_gguf_value(&mut file, value)?;
        }

        // Tensor descriptors (compute offsets)
        let mut offset: u64 = 0;
        for (tensor_info, data) in meta.tensors.iter().zip(tensor_data.iter()) {
            write_gguf_string(&mut file, &tensor_info.name)?;
            file.write_all(&(tensor_info.dims.len() as u32).to_le_bytes())?;
            for &dim in &tensor_info.dims {
                file.write_all(&dim.to_le_bytes())?;
            }
            file.write_all(&tensor_info.dtype.to_u32().to_le_bytes())?;
            file.write_all(&offset.to_le_bytes())?;

            // Update offset for next tensor (with 32-byte alignment)
            offset += data.len() as u64;
            offset = (offset + 31) & !31;
        }

        // Pad to 32-byte alignment for data section
        let pos = file.stream_position()? as usize;
        let aligned = (pos + 31) & !31;
        let padding = aligned - pos;
        file.write_all(&vec![0u8; padding])?;

        // Tensor data
        for data in tensor_data {
            file.write_all(data)?;
            // Pad to 32-byte alignment
            let pad = ((data.len() + 31) & !31) - data.len();
            if pad > 0 {
                file.write_all(&vec![0u8; pad])?;
            }
        }

        Ok(())
    }
}

// ============================================================
// I/O helpers
// ============================================================

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Maximum allowed GGUF string length: 16 MB.
const MAX_GGUF_STRING_LEN: u64 = 16_777_216;

fn read_gguf_string(r: &mut impl Read) -> io::Result<String> {
    let len_u64 = read_u64(r)?;
    if len_u64 > MAX_GGUF_STRING_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF string length {} exceeds maximum allowed {}",
                len_u64, MAX_GGUF_STRING_LEN
            ),
        ));
    }
    let len = usize::try_from(len_u64).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF string length does not fit usize",
        )
    })?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_gguf_value(r: &mut impl Read) -> io::Result<GgufValue> {
    let vtype = read_u32(r)?;
    match vtype {
        0 => Ok(GgufValue::U8({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0]
        })),
        1 => Ok(GgufValue::I8({
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            b[0] as i8
        })),
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufValue::U16(u16::from_le_bytes(b)))
        }
        3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufValue::I16(i16::from_le_bytes(b)))
        }
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufValue::Bool(b[0] != 0))
        }
        8 => Ok(GgufValue::String(read_gguf_string(r)?)),
        9 => {
            let arr_type = read_u32(r)?;
            let arr_len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(arr_len);
            for _ in 0..arr_len {
                // Read array elements based on element type
                let val = match arr_type {
                    4 => GgufValue::U32(read_u32(r)?),
                    6 => GgufValue::F32(read_f32(r)?),
                    8 => GgufValue::String(read_gguf_string(r)?),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!("unsupported array element type: {}", arr_type),
                        ))
                    }
                };
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(GgufValue::I64(i64::from_le_bytes(b)))
        }
        12 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(GgufValue::F64(f64::from_le_bytes(b)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported GGUF value type: {}", vtype),
        )),
    }
}

fn write_gguf_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn write_gguf_value(w: &mut impl Write, v: &GgufValue) -> io::Result<()> {
    match v {
        GgufValue::U32(val) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())
        }
        GgufValue::I32(val) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())
        }
        GgufValue::F32(val) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())
        }
        GgufValue::String(val) => {
            w.write_all(&8u32.to_le_bytes())?;
            write_gguf_string(w, val)
        }
        GgufValue::Bool(val) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[if *val { 1u8 } else { 0u8 }])
        }
        GgufValue::U64(val) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&val.to_le_bytes())
        }
        _ => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "write not implemented for this value type",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_path(name: &str) -> PathBuf {
        PathBuf::from("/tmp/claude-1000/-home-habitat-ternary-clawd/95e7afdf-b472-41a0-a3d5-73532dc4ecb7/scratchpad")
            .join(name)
    }

    #[test]
    fn test_gguf_write_read_roundtrip() {
        let path = test_path("test_roundtrip.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Create test weights
        let rows = 64;
        let cols = 128;
        let gs = 128;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();

        let pw = PlanarWeights::from_row_major(&weights, rows, cols, gs);

        // Write
        let mut writer = GgufWriter::new();
        writer.add_metadata("model.name", GgufValue::String("test".to_string()));
        writer.add_metadata("model.group_size", GgufValue::U32(gs as u32));
        writer.add_ternary_tensor("weight.0", &pw);
        writer.write(&path).unwrap();

        // Read back
        let gguf = GgufFile::open(&path).unwrap();

        // Check metadata
        assert!(gguf.metadata.contains_key("model.name"));
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "weight.0");
        assert_eq!(gguf.tensors[0].dims, vec![64, 128]);
        assert_eq!(gguf.tensors[0].dtype, GGUF_TYPE_Q1_58);

        // Load back as PlanarWeights
        let pw2 = gguf.load_planar_weights("weight.0", gs).unwrap();
        assert_eq!(pw2.rows, rows);
        assert_eq!(pw2.cols, cols);

        // Verify packed data matches
        let kp = cols / 4;
        for r in 0..rows {
            for c in 0..kp {
                assert_eq!(
                    pw.data[r * kp + c],
                    pw2.data[r * kp + c],
                    "data mismatch at r={}, c={}",
                    r,
                    c
                );
            }
        }

        // Verify scales match
        let gprow = cols / gs;
        for r in 0..rows {
            for g in 0..gprow {
                assert!(
                    (pw.scales_rm[r * gprow + g] - pw2.scales_rm[r * gprow + g]).abs() < 1e-7,
                    "scale mismatch at r={}, g={}",
                    r,
                    g
                );
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_f32_tensor() {
        let path = test_path("test_f32.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut writer = GgufWriter::new();
        writer.add_f32_tensor("test_tensor", &[2, 3], &data);
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "test_tensor");
        assert_eq!(gguf.tensors[0].dims, vec![2, 3]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_all_metadata_types() {
        // Build a buffer with various metadata types and read them back
        let path = test_path("test_meta_types.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut writer = GgufWriter::new();
        writer.add_metadata("key_u32", GgufValue::U32(42));
        writer.add_metadata("key_str", GgufValue::String("hello".to_string()));
        writer.add_metadata("key_u64", GgufValue::U64(9999999));
        writer.add_ternary_tensor(
            "w",
            &PlanarWeights::from_row_major(&vec![1.0f32; 4 * 128], 4, 128, 128),
        );
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("key_u32") {
            Some(GgufValue::U32(v)) => assert_eq!(*v, 42),
            _ => panic!("key_u32 not found or wrong type"),
        }
        match gguf.metadata.get("key_str") {
            Some(GgufValue::String(v)) => assert_eq!(v, "hello"),
            _ => panic!("key_str not found or wrong type"),
        }
        match gguf.metadata.get("key_u64") {
            Some(GgufValue::U64(v)) => assert_eq!(*v, 9999999),
            _ => panic!("key_u64 not found or wrong type"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let path = test_path("test_bad_magic.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Write file with bad magic
        let mut f = File::create(&path).unwrap();
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap(); // version
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_tensors
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata
        drop(f);

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("magic"),
            "error should mention magic: {}",
            err_msg
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_bad_version() {
        let path = test_path("test_bad_version.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut f = File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&99u32.to_le_bytes()).unwrap(); // bad version
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        drop(f);

        let result = GgufFile::open(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_planar_not_found() {
        let path = test_path("test_notfound.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut writer = GgufWriter::new();
        writer.add_metadata("model.group_size", GgufValue::U32(128));
        writer.add_ternary_tensor(
            "w0",
            &PlanarWeights::from_row_major(&vec![1.0f32; 4 * 128], 4, 128, 128),
        );
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let result = gguf.load_planar_weights("nonexistent", 128);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_writer_default() {
        let w = GgufWriter::default();
        assert!(w.tensors.is_empty());
        assert!(w.metadata.is_empty());
    }

    #[test]
    fn test_gguf_multiple_tensors() {
        let path = test_path("test_multi_tensor.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut writer = GgufWriter::new();
        writer.add_metadata("model.group_size", GgufValue::U32(128));
        writer.add_ternary_tensor(
            "w0",
            &PlanarWeights::from_row_major(&vec![1.0f32; 4 * 128], 4, 128, 128),
        );
        writer.add_ternary_tensor(
            "w1",
            &PlanarWeights::from_row_major(&vec![-1.0f32; 8 * 256], 8, 256, 128),
        );
        writer.add_f32_tensor("norm", &[4], &[1.0, 1.0, 1.0, 1.0]);
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 3);

        let pw0 = gguf.load_planar_weights("w0", 128).unwrap();
        assert_eq!(pw0.rows, 4);
        assert_eq!(pw0.cols, 128);

        let pw1 = gguf.load_planar_weights("w1", 128).unwrap();
        assert_eq!(pw1.rows, 8);
        assert_eq!(pw1.cols, 256);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_value_variants() {
        // Test Debug formatting for GgufValue variants
        let values = vec![
            GgufValue::U8(1),
            GgufValue::I8(-1),
            GgufValue::U16(100),
            GgufValue::I16(-100),
            GgufValue::U32(1000),
            GgufValue::I32(-1000),
            GgufValue::F32(std::f32::consts::PI),
            GgufValue::U64(99999),
            GgufValue::I64(-99999),
            GgufValue::F64(std::f64::consts::E),
            GgufValue::Bool(true),
            GgufValue::String("test".to_string()),
            GgufValue::Array(vec![GgufValue::U32(1)]),
        ];
        for v in &values {
            let _ = format!("{:?}", v);
        }
    }

    #[test]
    fn test_tensor_descriptor_debug() {
        let td = TensorDescriptor {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![4, 128],
            dtype: GGUF_TYPE_Q1_58,
            offset: 0,
        };
        let _ = format!("{:?}", td);
    }

    #[test]
    fn test_gguf_write_i32_f32_metadata() {
        let path = test_path("test_i32_f32_meta.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut writer = GgufWriter::new();
        writer.add_metadata("int_val", GgufValue::I32(-42));
        writer.add_metadata("float_val", GgufValue::F32(std::f32::consts::PI));
        writer.add_f32_tensor("t", &[2], &[1.0, 2.0]);
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("int_val") {
            Some(GgufValue::I32(v)) => assert_eq!(*v, -42),
            _ => panic!("int_val wrong"),
        }
        match gguf.metadata.get("float_val") {
            Some(GgufValue::F32(v)) => assert!((*v - std::f32::consts::PI).abs() < 0.01),
            _ => panic!("float_val wrong"),
        }

        std::fs::remove_file(&path).ok();
    }

    /// Helper: write a GGUF file with a single metadata key using raw binary encoding.
    /// This lets us test read_gguf_value branches for types the writer doesn't support.
    fn write_raw_gguf_with_meta(path: &std::path::Path, key: &str, vtype: u32, value_bytes: &[u8]) {
        let mut f = File::create(path).unwrap();
        // header
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_tensors=0
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_metadata=1
                                                   // metadata key (GGUF string: u64 len + bytes)
        write_gguf_string(&mut f, key).unwrap();
        // value type + data
        f.write_all(&vtype.to_le_bytes()).unwrap();
        f.write_all(value_bytes).unwrap();
    }

    #[test]
    fn test_gguf_read_u8_value() {
        let path = test_path("test_u8_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 0, &[42u8]);
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::U8(v)) => assert_eq!(*v, 42),
            other => panic!("expected U8(42), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_i8_value() {
        let path = test_path("test_i8_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 1, &[0xFEu8]); // -2 as i8
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::I8(v)) => assert_eq!(*v, -2),
            other => panic!("expected I8(-2), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_u16_value() {
        let path = test_path("test_u16_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 2, &1234u16.to_le_bytes());
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::U16(v)) => assert_eq!(*v, 1234),
            other => panic!("expected U16(1234), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_i16_value() {
        let path = test_path("test_i16_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 3, &(-500i16).to_le_bytes());
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::I16(v)) => assert_eq!(*v, -500),
            other => panic!("expected I16(-500), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_bool_value() {
        let path = test_path("test_bool_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 7, &[1u8]);
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::Bool(v)) => assert!(*v),
            other => panic!("expected Bool(true), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_i64_value() {
        let path = test_path("test_i64_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 11, &(-99999i64).to_le_bytes());
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::I64(v)) => assert_eq!(*v, -99999),
            other => panic!("expected I64(-99999), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_f64_value() {
        let path = test_path("test_f64_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 12, &(std::f64::consts::PI).to_le_bytes());
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::F64(v)) => assert!((*v - std::f64::consts::PI).abs() < 1e-10),
            other => panic!("expected F64, got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_array_u32_value() {
        let path = test_path("test_array_val.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Array type=9, element_type=U32(4), len=3, then 3 U32 values
        let mut val_bytes = Vec::new();
        val_bytes.extend_from_slice(&4u32.to_le_bytes()); // arr_type = U32
        val_bytes.extend_from_slice(&3u64.to_le_bytes()); // arr_len = 3
        val_bytes.extend_from_slice(&10u32.to_le_bytes());
        val_bytes.extend_from_slice(&20u32.to_le_bytes());
        val_bytes.extend_from_slice(&30u32.to_le_bytes());
        write_raw_gguf_with_meta(&path, "val", 9, &val_bytes);
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::Array(arr)) => {
                assert_eq!(arr.len(), 3);
                if let GgufValue::U32(v) = &arr[0] {
                    assert_eq!(*v, 10);
                }
                if let GgufValue::U32(v) = &arr[2] {
                    assert_eq!(*v, 30);
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_u64_value() {
        let path = test_path("test_u64_read.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 10, &99999u64.to_le_bytes());
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::U64(v)) => assert_eq!(*v, 99999),
            other => panic!("expected U64, got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_unsupported_value_type() {
        let path = test_path("test_bad_vtype.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_raw_gguf_with_meta(&path, "val", 255, &[0u8; 4]);
        let result = GgufFile::open(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_write_bool_roundtrip() {
        let path = test_path("test_write_bool.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut writer = GgufWriter::new();
        writer.add_metadata("flag_true", GgufValue::Bool(true));
        writer.add_metadata("flag_false", GgufValue::Bool(false));
        writer.add_f32_tensor("t", &[1], &[1.0]);
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("flag_true") {
            Some(GgufValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {:?}", other),
        }
        match gguf.metadata.get("flag_false") {
            Some(GgufValue::Bool(false)) => {}
            other => panic!("expected Bool(false), got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_non_2d_tensor() {
        let path = test_path("test_non2d.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut writer = GgufWriter::new();
        writer.add_metadata("model.group_size", GgufValue::U32(128));
        // f32 tensor is 1D, not 2D
        writer.add_f32_tensor("w", &[4], &[1.0, 2.0, 3.0, 4.0]);
        writer.write(&path).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        // Attempting to load_planar_weights on a 1D tensor should fail
        let result = gguf.load_planar_weights("w", 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("2D"), "error: {}", err);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_array_f32() {
        let path = test_path("test_array_f32.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Array type=9, element_type=F32(6), len=2
        let mut val_bytes = Vec::new();
        val_bytes.extend_from_slice(&6u32.to_le_bytes()); // arr_type = F32
        val_bytes.extend_from_slice(&2u64.to_le_bytes()); // arr_len = 2
        val_bytes.extend_from_slice(&1.5f32.to_le_bytes());
        val_bytes.extend_from_slice(&2.5f32.to_le_bytes());
        write_raw_gguf_with_meta(&path, "val", 9, &val_bytes);
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::Array(arr)) => {
                assert_eq!(arr.len(), 2);
                if let GgufValue::F32(v) = &arr[0] {
                    assert!((*v - 1.5).abs() < 1e-6);
                }
                if let GgufValue::F32(v) = &arr[1] {
                    assert!((*v - 2.5).abs() < 1e-6);
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_array_string() {
        let path = test_path("test_array_str.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Array type=9, element_type=String(8), len=1
        let mut val_bytes = Vec::new();
        val_bytes.extend_from_slice(&8u32.to_le_bytes()); // arr_type = String
        val_bytes.extend_from_slice(&1u64.to_le_bytes()); // arr_len = 1
                                                          // String: u64 len + bytes
        let s = "hello";
        val_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        val_bytes.extend_from_slice(s.as_bytes());
        write_raw_gguf_with_meta(&path, "val", 9, &val_bytes);
        let gguf = GgufFile::open(&path).unwrap();
        match gguf.metadata.get("val") {
            Some(GgufValue::Array(arr)) => {
                assert_eq!(arr.len(), 1);
                if let GgufValue::String(v) = &arr[0] {
                    assert_eq!(v, "hello");
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_read_array_unsupported_element() {
        let path = test_path("test_array_bad.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Array type=9, element_type=99 (unsupported), len=1
        let mut val_bytes = Vec::new();
        val_bytes.extend_from_slice(&99u32.to_le_bytes()); // arr_type = unsupported
        val_bytes.extend_from_slice(&1u64.to_le_bytes()); // arr_len = 1
        val_bytes.extend_from_slice(&[0u8; 4]); // dummy data
        write_raw_gguf_with_meta(&path, "val", 9, &val_bytes);
        let result = GgufFile::open(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_tensor_byte_size_f16() {
        let path = test_path("test_f16_size.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Manually write a GGUF with a tensor that has dtype=1 (F16)
        let mut f = File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors = 1
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata = 0
                                                   // tensor descriptor: name
        write_gguf_string(&mut f, "t").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // n_dims = 1
        f.write_all(&4u64.to_le_bytes()).unwrap(); // dim[0] = 4
        f.write_all(&1u32.to_le_bytes()).unwrap(); // dtype = 1 (F16)
        f.write_all(&0u64.to_le_bytes()).unwrap(); // offset = 0
                                                   // Pad to 32-byte alignment
        let pos = f.stream_position().unwrap() as usize;
        let pad = ((pos + 31) & !31) - pos;
        f.write_all(&vec![0u8; pad]).unwrap();
        // Write 4 F16 values (8 bytes)
        f.write_all(&[0u8; 8]).unwrap();
        drop(f);

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        let data = gguf.tensor_data(&gguf.tensors[0]).unwrap();
        assert_eq!(data.len(), 8); // 4 * 2 bytes for F16

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_tensor_byte_size_unknown_dtype() {
        let path = test_path("test_unk_dtype.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut f = File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors = 1
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata = 0
        write_gguf_string(&mut f, "t").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // n_dims = 1
        f.write_all(&4u64.to_le_bytes()).unwrap(); // dim[0] = 4
        f.write_all(&99u32.to_le_bytes()).unwrap(); // dtype = 99 (unknown)
        f.write_all(&0u64.to_le_bytes()).unwrap(); // offset = 0
        let pos = f.stream_position().unwrap() as usize;
        let pad = ((pos + 31) & !31) - pos;
        f.write_all(&vec![0u8; pad]).unwrap();
        f.write_all(&[0u8; 4]).unwrap(); // 4 bytes of data
        drop(f);

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        // Unknown dtype should now return an error
        let result = gguf.tensor_data(&gguf.tensors[0]);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("unsupported GGUF dtype"),
            "unexpected error: {}",
            err_msg
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_tensor_byte_size_no_group_size_meta() {
        let path = test_path("test_no_gs.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Write a GGUF with Q1_58 tensor but NO model.group_size metadata
        let mut f = File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors = 1
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata = 0
        write_gguf_string(&mut f, "w").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims = 2
        f.write_all(&4u64.to_le_bytes()).unwrap(); // dim[0] = 4 (rows)
        f.write_all(&128u64.to_le_bytes()).unwrap(); // dim[1] = 128 (cols)
        f.write_all(&GGUF_TYPE_Q1_58.to_le_bytes()).unwrap(); // dtype = Q1_58
        f.write_all(&0u64.to_le_bytes()).unwrap(); // offset = 0
        let pos = f.stream_position().unwrap() as usize;
        let pad = ((pos + 31) & !31) - pos;
        f.write_all(&vec![0u8; pad]).unwrap();
        // packed bytes: 4 rows * 32 bytes = 128 bytes
        // scales: 4 rows * 1 group * 4 bytes = 16 bytes
        f.write_all(&[0u8; 128 + 16]).unwrap();
        drop(f);

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.tensors.len(), 1);
        let err = gguf.tensor_data(&gguf.tensors[0]).unwrap_err().to_string();
        assert!(
            err.contains("missing group_size metadata"),
            "unexpected error: {}",
            err
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_load_planar_data_too_short() {
        let path = test_path("test_short_data.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Write a GGUF with Q1_58 tensor but truncated data
        let mut f = File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&GGUF_VERSION.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors = 1
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_metadata = 1
                                                   // metadata: model.group_size = 128
        write_gguf_string(&mut f, "model.group_size").unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap(); // type = U32
        f.write_all(&128u32.to_le_bytes()).unwrap();
        // tensor descriptor
        write_gguf_string(&mut f, "w").unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims = 2
        f.write_all(&4u64.to_le_bytes()).unwrap(); // rows = 4
        f.write_all(&128u64.to_le_bytes()).unwrap(); // cols = 128
        f.write_all(&GGUF_TYPE_Q1_58.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // offset = 0
        let pos = f.stream_position().unwrap() as usize;
        let pad = ((pos + 31) & !31) - pos;
        f.write_all(&vec![0u8; pad]).unwrap();
        // Write only 10 bytes (far too short: need 128 + 16 = 144)
        f.write_all(&[0u8; 10]).unwrap();
        drop(f);

        let gguf = GgufFile::open(&path).unwrap();
        let result = gguf.load_planar_weights("w", 128);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("too short") || err.contains("exceeds file size"),
            "error: {}",
            err
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_f32_tensor_data() {
        let path = test_path("test_f32_data.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut writer = GgufWriter::new();
        writer.add_f32_tensor("norm", &[4], &[1.0, 2.0, 3.0, 4.0]);
        writer.write(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let data = gguf.tensor_data(&gguf.tensors[0]).unwrap();
        // 4 F32 values = 16 bytes
        assert_eq!(data.len(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_debug_format() {
        let path = test_path("test_debug_fmt.gguf");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut writer = GgufWriter::new();
        writer.add_metadata("key", GgufValue::U32(1));
        writer.add_f32_tensor("t", &[1], &[1.0]);
        writer.write(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let _ = format!("{:?}", gguf);
        std::fs::remove_file(&path).ok();
    }
}
