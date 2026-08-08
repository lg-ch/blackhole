-- Pre-create the typical metadata table mangrove ClickHouseSink expects.
-- Customize the column list to match your application.
CREATE TABLE IF NOT EXISTS docs_metadata (
    doc_id    UInt32,
    ts        DateTime64(3) DEFAULT now64(),
    category  LowCardinality(String),
    region    LowCardinality(String),
    lang      LowCardinality(String)
) ENGINE = MergeTree
ORDER BY (ts, doc_id);
