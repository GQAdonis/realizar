//! GGUF (GPT-Generated Unified Format) parser
//!
//! Pure Rust implementation of GGUF binary format reader.
//! Used by llama.cpp, Ollama, and compatible tools.
//!
//! Format specification: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! ## Format Overview
//!
//! ```text
//! GGUF := HEADER METADATA[] TENSOR_INFO[] TENSOR_DATA[]
//!
//! HEADER := {
//!   magic: u32 = 0x46554747 ("GGUF")
//!   version: u32
//!   tensor_count: u64
//!   metadata_count: u64
//! }
//! ```

use std::{
    collections::HashMap,
    io::{Cursor, Read},
};

use crate::error::{RealizarError, Result};

/// GGUF magic number: "GGUF" in little-endian
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF versions
pub const GGUF_VERSION_V3: u32 = 3;

/// GGUF metadata value types
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    /// Unsigned 8-bit integer
    UInt8(u8),
    /// Signed 8-bit integer
    Int8(i8),
    /// Unsigned 16-bit integer
    UInt16(u16),
    /// Signed 16-bit integer
    Int16(i16),
    /// Unsigned 32-bit integer
    UInt32(u32),
    /// Signed 32-bit integer
    Int32(i32),
    /// 32-bit floating point
    Float32(f32),
    /// Boolean
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values
    Array(Vec<GGUFValue>),
    /// Unsigned 64-bit integer
    UInt64(u64),
    /// Signed 64-bit integer
    Int64(i64),
    /// 64-bit floating point
    Float64(f64),
}

/// GGUF file header
#[derive(Debug, Clone, PartialEq)]
pub struct GGUFHeader {
    /// Magic number (must be `GGUF_MAGIC`)
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_count: u64,
}

/// Tensor information
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions (shape)
    pub dims: Vec<u64>,
    /// Quantization type
    pub qtype: u32,
    /// Offset in the file where tensor data starts
    pub offset: u64,
}

/// GGUF model container
#[derive(Debug, Clone)]
pub struct GGUFModel {
    /// File header
    pub header: GGUFHeader,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, GGUFValue>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
}

impl GGUFModel {
    /// Parse GGUF file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw GGUF file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid magic number
    /// - Unsupported version
    /// - Malformed data
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header
        let header = Self::parse_header(&mut cursor)?;

        // Parse metadata
        let metadata = Self::parse_metadata(&mut cursor, header.metadata_count);

        // Parse tensor info
        let tensors = Self::parse_tensor_info(&mut cursor, header.tensor_count);

        Ok(Self {
            header,
            metadata,
            tensors,
        })
    }

    /// Parse GGUF header
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<GGUFHeader> {
        let mut buf = [0u8; 4];

        // Read magic
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_magic".to_string(),
                reason: e.to_string(),
            })?;
        let magic = u32::from_le_bytes(buf);

        if magic != GGUF_MAGIC {
            return Err(RealizarError::InvalidShape {
                reason: format!("Invalid GGUF magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            });
        }

        // Read version
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_version".to_string(),
                reason: e.to_string(),
            })?;
        let version = u32::from_le_bytes(buf);

        if version != GGUF_VERSION_V3 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_gguf".to_string(),
                reason: format!("Unsupported GGUF version: {version}, only v3 supported"),
            });
        }

        // Read tensor_count
        let mut buf8 = [0u8; 8];
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_tensor_count".to_string(),
                reason: e.to_string(),
            })?;
        let tensor_count = u64::from_le_bytes(buf8);

        // Read metadata_count
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_count".to_string(),
                reason: e.to_string(),
            })?;
        let metadata_count = u64::from_le_bytes(buf8);

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_count,
        })
    }

    /// Parse metadata (stub - to be implemented)
    fn parse_metadata(_cursor: &mut Cursor<&[u8]>, _count: u64) -> HashMap<String, GGUFValue> {
        // TODO: Implement metadata parsing
        HashMap::new()
    }

    /// Parse tensor info (stub - to be implemented)
    fn parse_tensor_info(_cursor: &mut Cursor<&[u8]>, _count: u64) -> Vec<TensorInfo> {
        // TODO: Implement tensor info parsing
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic_constant() {
        // "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        // Verify it spells "GGUF"
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn test_parse_valid_header() {
        // Minimal valid GGUF v3 header
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

        let model = GGUFModel::from_bytes(&data).unwrap();
        assert_eq!(model.header.magic, GGUF_MAGIC);
        assert_eq!(model.header.version, 3);
        assert_eq!(model.header.tensor_count, 0);
        assert_eq!(model.header.metadata_count, 0);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"BAAD"); // Invalid magic
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::InvalidShape { .. }
        ));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&999u32.to_le_bytes()); // Unsupported version
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GGUFModel::from_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::UnsupportedOperation { .. }
        ));
    }

    #[test]
    fn test_truncated_data() {
        // Only 4 bytes (magic only)
        let data = b"GGUF";
        let result = GGUFModel::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let data = &[];
        let result = GGUFModel::from_bytes(data);
        assert!(result.is_err());
    }
}
