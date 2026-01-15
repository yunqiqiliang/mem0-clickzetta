# ClickZetta vs 其他向量库的表结构对比

## 📊 向量库表结构对比分析

### 🔍 ClickZetta 表结构

```sql
CREATE TABLE IF NOT EXISTS collection_name (
    id String,                                    -- 主键，字符串类型
    content String,                               -- 内容字段
    metadata String,                              -- 元数据，JSON字符串
    embedding vector(Float32, 1536)              -- 向量字段，指定维度
)

-- 向量索引
CREATE INDEX embedding_idx_xxx ON collection_name (embedding) Vector
```

### 🔍 PostgreSQL (pgvector) 表结构

```sql
CREATE TABLE IF NOT EXISTS collection_name (
    id UUID PRIMARY KEY,                          -- UUID主键，有主键约束
    vector vector(1536),                          -- 向量字段
    payload JSONB                                 -- 元数据，原生JSONB类型
);

-- 向量索引 (可选)
CREATE INDEX collection_name_hnsw_idx
ON collection_name USING hnsw (vector vector_cosine_ops);
```

### 🔍 Azure MySQL 表结构

```sql
CREATE TABLE IF NOT EXISTS collection_name (
    id VARCHAR(255) PRIMARY KEY,                  -- 字符串主键，有主键约束
    vector JSON,                                  -- 向量存储为JSON
    payload JSON,                                 -- 元数据，原生JSON类型
    INDEX idx_payload_keys ((CAST(payload AS CHAR(255)) ARRAY))
);
```

### 🔍 MongoDB 结构 (文档数据库)

```javascript
// 文档结构
{
    _id: ObjectId,                                // MongoDB默认主键
    vector: [0.1, 0.2, 0.3, ...],               // 向量数组
    payload: {                                    // 元数据对象
        data: "content",
        user_id: "user1",
        // ... 其他字段
    }
}

// 向量搜索索引
db.collection.createSearchIndex({
    name: "vector_index",
    type: "vectorSearch",
    definition: {
        fields: [{
            type: "vector",
            path: "vector",
            numDimensions: 1536,
            similarity: "cosine"
        }]
    }
})
```

## 📋 详细对比表

| 特性 | ClickZetta | PostgreSQL | Azure MySQL | MongoDB |
|------|------------|------------|-------------|---------|
| **主键类型** | String (无约束) | UUID (PRIMARY KEY) | VARCHAR(255) (PRIMARY KEY) | ObjectId (自动) |
| **主键约束** | ❌ 无 | ✅ 有 | ✅ 有 | ✅ 有 |
| **向量存储** | `vector(Float32, dims)` | `vector(dims)` | `JSON` | `Array` |
| **元数据存储** | `String` (JSON字符串) | `JSONB` (原生) | `JSON` (原生) | `Object` (原生) |
| **内容字段** | ✅ 独立 `content` 字段 | ❌ 存储在 payload 中 | ❌ 存储在 payload 中 | ❌ 存储在 payload 中 |
| **向量索引** | Vector Index | HNSW/IVFFlat | 无专用向量索引 | Vector Search Index |
| **UPSERT支持** | ❌ 需要手动实现 | ✅ ON CONFLICT | ✅ ON DUPLICATE KEY | ✅ 原生支持 |

## 🔍 关键差异分析

### 1. **主键约束差异**

**ClickZetta 的独特之处**：
- 没有 PRIMARY KEY 约束
- 允许重复的 ID（这是重复记录问题的根源）
- 需要手动实现唯一性检查

**其他数据库**：
- 都有主键约束，自动防止重复
- 插入重复 ID 会报错或触发 UPSERT

### 2. **字段结构差异**

**ClickZetta 的设计**：
```
id (String) + content (String) + metadata (String) + embedding (Vector)
```

**标准设计**：
```
id (Primary Key) + payload (JSON/JSONB) + vector (Vector)
```

### 3. **数据存储方式**

| 数据库 | 内容存储位置 | 元数据存储 |
|--------|-------------|-----------|
| **ClickZetta** | 独立 `content` 字段 | JSON 字符串 |
| **PostgreSQL** | `payload.data` | 原生 JSONB |
| **MySQL** | `payload.data` | 原生 JSON |
| **MongoDB** | `payload.data` | 原生对象 |

## 🎯 ClickZetta 设计的优缺点

### ✅ **优点**

1. **明确的字段分离**：内容和元数据分开存储
2. **向量类型支持**：原生向量类型，性能更好
3. **灵活的索引**：专门的向量索引支持

### ❌ **缺点**

1. **缺少主键约束**：需要手动实现重复检查
2. **元数据处理复杂**：JSON 字符串需要手动解析
3. **不符合标准模式**：与其他向量库差异较大

## 🛠️ 我们的解决方案

为了让 ClickZetta 与其他向量库保持一致，我们实现了：

1. **手动 UPSERT**：检查 + 插入/更新
2. **数据格式统一**：在 `get()` 方法中重构数据格式
3. **兼容性处理**：支持 `data`、`content`、`text` 字段

## 💡 建议

如果要让 ClickZetta 更符合标准，可以考虑：

1. **添加主键约束**：
```sql
CREATE TABLE collection_name (
    id String PRIMARY KEY,  -- 添加主键约束
    -- ...
)
```

2. **简化字段结构**：
```sql
CREATE TABLE collection_name (
    id String PRIMARY KEY,
    payload String,         -- 合并 content 和 metadata
    embedding vector(Float32, 1536)
)
```

但这需要修改现有的实现逻辑。目前的手动 UPSERT 方案是一个很好的兼容性解决方案。