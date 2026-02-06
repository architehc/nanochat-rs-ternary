// io.rs — Binary serialization for mHC parameters
//
// File format:
//   Header: [magic:u32 = 0x6D484321][version:u32 = 1][n_layers:u32][n_streams:u32]
//   Per layer (N=2): 36 bytes
//   Per layer (N=4): 160 bytes

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::n2::MhcLiteN2;
use crate::n4::MhcLiteN4;

/// Magic number: "mHC!" in little-endian
pub const MHC_MAGIC: u32 = 0x6D484321;
pub const MHC_VERSION: u32 = 1;

/// File header for mHC binary format.
#[derive(Debug, Clone)]
pub struct MhcFileHeader {
    pub magic: u32,
    pub version: u32,
    pub n_layers: u32,
    pub n_streams: u32,
}

/// Parameters for a single mHC layer (either N=2 or N=4).
#[derive(Debug, Clone)]
pub enum MhcLayerParams {
    N2(MhcLiteN2),
    N4(MhcLiteN4),
}

/// Load mHC parameters from a binary file.
pub fn load_mhc_file<P: AsRef<Path>>(path: P) -> io::Result<(MhcFileHeader, Vec<MhcLayerParams>)> {
    let mut file = File::open(path)?;
    let mut buf4 = [0u8; 4];

    // Read header
    file.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != MHC_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid mHC magic: {:#010x}", magic)));
    }

    file.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != MHC_VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported mHC version: {}", version)));
    }

    file.read_exact(&mut buf4)?;
    let n_layers = u32::from_le_bytes(buf4);

    file.read_exact(&mut buf4)?;
    let n_streams = u32::from_le_bytes(buf4);

    let header = MhcFileHeader {
        magic,
        version,
        n_layers,
        n_streams,
    };

    // Read layer data
    let bytes_per_layer = match n_streams {
        2 => 36,
        4 => 160,
        _ => { return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported n_streams: {}", n_streams))); }
    };

    let mut layers = Vec::with_capacity(n_layers as usize);
    for _ in 0..n_layers {
        let mut buf = vec![0u8; bytes_per_layer];
        file.read_exact(&mut buf)?;

        let params = match n_streams {
            2 => MhcLayerParams::N2(
                MhcLiteN2::from_bytes(&buf)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad N2 data"))?,
            ),
            4 => MhcLayerParams::N4(
                MhcLiteN4::from_bytes(&buf)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad N4 data"))?,
            ),
            _ => unreachable!(),
        };
        layers.push(params);
    }

    Ok((header, layers))
}

/// Save mHC parameters to a binary file.
pub fn save_mhc_file<P: AsRef<Path>>(
    path: P,
    n_streams: u32,
    layers: &[MhcLayerParams],
) -> io::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    file.write_all(&MHC_MAGIC.to_le_bytes())?;
    file.write_all(&MHC_VERSION.to_le_bytes())?;
    file.write_all(&(layers.len() as u32).to_le_bytes())?;
    file.write_all(&n_streams.to_le_bytes())?;

    // Write layer data
    for layer in layers {
        match layer {
            MhcLayerParams::N2(mhc) => file.write_all(&mhc.to_bytes())?,
            MhcLayerParams::N4(mhc) => file.write_all(&mhc.to_bytes())?,
        }
    }

    Ok(())
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
    fn test_save_load_n2() {
        let path = test_path("test_mhc_n2.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let layers = vec![
            MhcLayerParams::N2(MhcLiteN2::new_identity()),
            MhcLayerParams::N2(MhcLiteN2::from_weights(
                2.0,
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
            )),
        ];

        save_mhc_file(&path, 2, &layers).unwrap();
        let (header, loaded) = load_mhc_file(&path).unwrap();

        assert_eq!(header.magic, MHC_MAGIC);
        assert_eq!(header.version, MHC_VERSION);
        assert_eq!(header.n_layers, 2);
        assert_eq!(header.n_streams, 2);
        assert_eq!(loaded.len(), 2);

        if let MhcLayerParams::N2(mhc) = &loaded[1] {
            assert!((mhc.alpha_logit - 2.0).abs() < 1e-7);
        } else {
            panic!("expected N2");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_n4() {
        let path = test_path("test_mhc_n4.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let layers = vec![
            MhcLayerParams::N4(MhcLiteN4::new_identity()),
            MhcLayerParams::N4(MhcLiteN4::new_identity()),
            MhcLayerParams::N4(MhcLiteN4::new_identity()),
        ];

        save_mhc_file(&path, 4, &layers).unwrap();
        let (header, loaded) = load_mhc_file(&path).unwrap();

        assert_eq!(header.n_layers, 3);
        assert_eq!(header.n_streams, 4);
        assert_eq!(loaded.len(), 3);

        if let MhcLayerParams::N4(mhc) = &loaded[0] {
            assert!((mhc.res_logits[0] - 10.0).abs() < 1e-7);
        } else {
            panic!("expected N4");
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let path = test_path("test_mhc_bad_magic.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Write file with wrong magic
        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        drop(f);

        let result = load_mhc_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("magic"), "error: {}", err);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_bad_version() {
        let path = test_path("test_mhc_bad_ver.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&MHC_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&99u32.to_le_bytes()).unwrap(); // bad version
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        drop(f);

        let result = load_mhc_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("version"), "error: {}", err);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_bad_n_streams() {
        let path = test_path("test_mhc_bad_streams.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        use std::io::Write;
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&MHC_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&MHC_VERSION.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // n_layers
        f.write_all(&7u32.to_le_bytes()).unwrap(); // unsupported n_streams
        drop(f);

        let result = load_mhc_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("n_streams"), "error: {}", err);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_mhc_layer_params_debug() {
        let n2 = MhcLayerParams::N2(MhcLiteN2::new_identity());
        let n4 = MhcLayerParams::N4(MhcLiteN4::new_identity());
        let _ = format!("{:?}", n2);
        let _ = format!("{:?}", n4);
    }

    #[test]
    fn test_mhc_file_header_clone() {
        let h = MhcFileHeader {
            magic: MHC_MAGIC,
            version: MHC_VERSION,
            n_layers: 3,
            n_streams: 2,
        };
        let h2 = h.clone();
        assert_eq!(h2.n_layers, 3);
        let _ = format!("{:?}", h2);
    }

    #[test]
    fn test_file_size() {
        let path = test_path("test_mhc_size.bin");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let layers_n2: Vec<_> = (0..4)
            .map(|_| MhcLayerParams::N2(MhcLiteN2::new_identity()))
            .collect();
        save_mhc_file(&path, 2, &layers_n2).unwrap();

        let size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(size, 16 + 4 * 36); // header + 4 * 36 bytes

        std::fs::remove_file(&path).ok();
    }
}
