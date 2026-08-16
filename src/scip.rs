use anyhow::{Result, bail, ensure};

#[derive(Debug, Default)]
pub(crate) struct ScipIndex {
    pub documents: Vec<ScipDocument>,
}

#[derive(Debug, Default)]
pub(crate) struct ScipDocument {
    pub relative_path: String,
    pub language: String,
    pub position_encoding: i32,
    pub occurrences: Vec<ScipOccurrence>,
}

#[derive(Debug, Default)]
pub(crate) struct ScipOccurrence {
    pub range: Option<ScipRange>,
    pub symbol: String,
    pub symbol_roles: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScipRange {
    pub start_line: u32,
    pub start_character: u32,
    pub end_line: u32,
    pub end_character: u32,
}

pub(crate) fn decode_index(bytes: &[u8]) -> Result<ScipIndex> {
    let mut reader = ProtoReader::new(bytes);
    let mut index = ScipIndex::default();
    while let Some((field, wire)) = reader.key()? {
        if field == 2 && wire == 2 {
            index.documents.push(decode_document(reader.bytes()?)?);
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(index)
}

fn decode_document(bytes: &[u8]) -> Result<ScipDocument> {
    let mut reader = ProtoReader::new(bytes);
    let mut document = ScipDocument::default();
    while let Some((field, wire)) = reader.key()? {
        match (field, wire) {
            (1, 2) => document.relative_path = reader.string()?,
            (2, 2) => document
                .occurrences
                .push(decode_occurrence(reader.bytes()?)?),
            (4, 2) => document.language = reader.string()?,
            (6, 0) => document.position_encoding = reader.varint()? as i32,
            _ => reader.skip(wire)?,
        }
    }
    Ok(document)
}

fn decode_occurrence(bytes: &[u8]) -> Result<ScipOccurrence> {
    let mut reader = ProtoReader::new(bytes);
    let mut occurrence = ScipOccurrence::default();
    let mut legacy_range = Vec::new();
    while let Some((field, wire)) = reader.key()? {
        match (field, wire) {
            (1, 2) => {
                let mut packed = ProtoReader::new(reader.bytes()?);
                while !packed.finished() {
                    legacy_range.push(packed.varint()? as u32);
                }
            }
            (1, 0) => legacy_range.push(reader.varint()? as u32),
            (2, 2) => occurrence.symbol = reader.string()?,
            (3, 0) => occurrence.symbol_roles = reader.varint()? as i32,
            (8, 2) => occurrence.range = Some(decode_single_line_range(reader.bytes()?)?),
            (9, 2) => occurrence.range = Some(decode_multi_line_range(reader.bytes()?)?),
            _ => reader.skip(wire)?,
        }
    }
    if occurrence.range.is_none() {
        occurrence.range = match legacy_range.as_slice() {
            [line, start, end] => Some(ScipRange {
                start_line: *line,
                start_character: *start,
                end_line: *line,
                end_character: *end,
            }),
            [start_line, start, end_line, end] => Some(ScipRange {
                start_line: *start_line,
                start_character: *start,
                end_line: *end_line,
                end_character: *end,
            }),
            _ => None,
        };
    }
    Ok(occurrence)
}

fn decode_single_line_range(bytes: &[u8]) -> Result<ScipRange> {
    let mut reader = ProtoReader::new(bytes);
    let mut line = 0;
    let mut start = 0;
    let mut end = 0;
    while let Some((field, wire)) = reader.key()? {
        match (field, wire) {
            (1, 0) => line = reader.varint()? as u32,
            (2, 0) => start = reader.varint()? as u32,
            (3, 0) => end = reader.varint()? as u32,
            _ => reader.skip(wire)?,
        }
    }
    Ok(ScipRange {
        start_line: line,
        start_character: start,
        end_line: line,
        end_character: end,
    })
}

fn decode_multi_line_range(bytes: &[u8]) -> Result<ScipRange> {
    let mut reader = ProtoReader::new(bytes);
    let mut range = ScipRange {
        start_line: 0,
        start_character: 0,
        end_line: 0,
        end_character: 0,
    };
    while let Some((field, wire)) = reader.key()? {
        match (field, wire) {
            (1, 0) => range.start_line = reader.varint()? as u32,
            (2, 0) => range.start_character = reader.varint()? as u32,
            (3, 0) => range.end_line = reader.varint()? as u32,
            (4, 0) => range.end_character = reader.varint()? as u32,
            _ => reader.skip(wire)?,
        }
    }
    Ok(range)
}

struct ProtoReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn key(&mut self) -> Result<Option<(u32, u8)>> {
        if self.finished() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        let wire = (key & 0x07) as u8;
        ensure!(field != 0, "SCIP protobuf contains field zero");
        Ok(Some((field, wire)))
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let Some(byte) = self.bytes.get(self.offset).copied() else {
                bail!("truncated SCIP protobuf varint");
            };
            self.offset += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("SCIP protobuf varint is too long")
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(self.varint()?)?;
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| anyhow::anyhow!("SCIP protobuf length overflow"))?;
        ensure!(end <= self.bytes.len(), "truncated SCIP protobuf field");
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn string(&mut self) -> Result<String> {
        Ok(std::str::from_utf8(self.bytes()?)?.to_owned())
    }

    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.advance(8)?,
            2 => {
                self.bytes()?;
            }
            5 => self.advance(4)?,
            _ => bail!("unsupported SCIP protobuf wire type {wire}"),
        }
        Ok(())
    }

    fn advance(&mut self, count: usize) -> Result<()> {
        self.offset = self
            .offset
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("SCIP protobuf offset overflow"))?;
        ensure!(
            self.offset <= self.bytes.len(),
            "truncated SCIP protobuf field"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(field: u8, payload: &[u8]) -> Vec<u8> {
        let mut encoded = vec![(field << 3) | 2, payload.len() as u8];
        encoded.extend_from_slice(payload);
        encoded
    }

    #[test]
    fn decodes_documents_occurrences_and_typed_ranges() {
        let typed_range = vec![8, 4, 16, 2, 24, 7];
        let mut occurrence = field(8, &typed_range);
        occurrence.extend(field(2, b"rust cargo demo 1.0 crate/run()."));
        occurrence.extend([24, 1]);
        let mut document = field(1, b"src/main.rs");
        document.extend(field(2, &occurrence));
        document.extend(field(4, b"rust"));
        document.extend([48, 1]);
        let index = decode_index(&field(2, &document)).unwrap();

        assert_eq!(index.documents.len(), 1);
        let document = &index.documents[0];
        assert_eq!(document.relative_path, "src/main.rs");
        assert_eq!(document.language, "rust");
        assert_eq!(document.position_encoding, 1);
        assert_eq!(document.occurrences[0].symbol_roles, 1);
        assert_eq!(
            document.occurrences[0].range,
            Some(ScipRange {
                start_line: 4,
                start_character: 2,
                end_line: 4,
                end_character: 7,
            })
        );
    }
}
