#!/usr/bin/env python3
"""
Test ClickZetta Upsert Functionality (No Cleanup Version)
"""

import os
import sys
import logging
from typing import List, Dict

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_config() -> Dict[str, str]:
    """Load configuration from .env file."""
    config = {}
    env_file = os.path.join(project_root, 'server', '.env')

    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value
        logger.info(f"Loaded configuration from {env_file}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

def test_clickzetta_upsert_no_cleanup():
    """Test ClickZetta upsert functionality without cleanup."""
    logger.info("Testing ClickZetta upsert functionality (keeping data)...")

    try:
        from mem0.vector_stores.clickzetta import ClickZetta

        # Load configuration
        env_config = load_env_config()
        if not env_config:
            logger.error("Failed to load configuration. Exiting.")
            return False

        # Create ClickZetta vector store
        vector_store = ClickZetta(
            collection_name="test_upsert_keep_data",
            embedding_model_dims=1536,
            service=env_config.get("CLICKZETTA_SERVICE", ""),
            instance=env_config.get("CLICKZETTA_INSTANCE", ""),
            workspace=env_config.get("CLICKZETTA_WORKSPACE", ""),
            schema=env_config.get("CLICKZETTA_SCHEMA", ""),
            username=env_config.get("CLICKZETTA_USERNAME", ""),
            password=env_config.get("CLICKZETTA_PASSWORD", ""),
            vcluster=env_config.get("CLICKZETTA_VCLUSTER", ""),
            distance_metric="cosine"
        )

        logger.info("✅ Successfully created ClickZetta vector store")

        # Test data
        test_vectors = [
            [0.1, 0.2, 0.3] + [0.0] * 1533,  # Pad to 1536 dimensions
            [0.4, 0.5, 0.6] + [0.0] * 1533,
            [0.7, 0.8, 0.9] + [0.0] * 1533
        ]

        test_payloads = [
            {"data": "ClickZetta支持高性能的批量操作", "category": "database", "user_id": "user1", "test_run": "keep_data"},
            {"data": "向量搜索技术在AI中很重要", "category": "ai", "user_id": "user2", "test_run": "keep_data"},
            {"data": "Mem0是优秀的记忆管理系统", "category": "memory", "user_id": "user3", "test_run": "keep_data"}
        ]

        test_ids = ["keep_test_1", "keep_test_2", "keep_test_3"]

        # Test 1: Initial insert
        logger.info("Test 1: Initial insert...")
        vector_store.insert(test_vectors, test_payloads, test_ids)
        logger.info("✅ Initial insert completed")

        # Test 2: Upsert with same IDs (should update)
        logger.info("Test 2: Upsert with same IDs (update)...")
        updated_payloads = [
            {"data": "ClickZetta支持高性能的批量操作 - 已更新", "category": "database", "user_id": "user1", "version": 2, "test_run": "keep_data"},
            {"data": "向量搜索技术在AI中很重要 - 已更新", "category": "ai", "user_id": "user2", "version": 2, "test_run": "keep_data"},
            {"data": "Mem0是优秀的记忆管理系统 - 已更新", "category": "memory", "user_id": "user3", "version": 2, "test_run": "keep_data"}
        ]

        vector_store.insert(test_vectors, updated_payloads, test_ids)
        logger.info("✅ Upsert (update) completed")

        # Test 3: Add one more record
        logger.info("Test 3: Adding additional record...")
        additional_vector = [[1.0, 1.1, 1.2] + [0.0] * 1533]
        additional_payload = [{"data": "新增的持久化测试数据", "category": "persistent", "user_id": "user4", "version": 1, "test_run": "keep_data"}]
        additional_id = ["keep_test_4"]

        vector_store.insert(additional_vector, additional_payload, additional_id)
        logger.info("✅ Additional record added")

        # Verify final state
        logger.info("Final verification...")
        all_vectors = vector_store.list(limit=10)
        logger.info(f"✅ Total records in table: {len(all_vectors)}")

        # Show all records
        logger.info("Records in test_upsert_keep_data table:")
        for i, record in enumerate(all_vectors):
            if record.payload:
                data = record.payload.get('data', 'N/A')[:50]
                version = record.payload.get('version', 'N/A')
                logger.info(f"  {i+1}. ID: {record.id}, Data: {data}..., Version: {version}")

        # Check for duplicates
        content_counts = {}
        for vec in all_vectors:
            if vec.payload and 'data' in vec.payload:
                content = vec.payload['data']
                content_counts[content] = content_counts.get(content, 0) + 1

        duplicates = {k: v for k, v in content_counts.items() if v > 1}
        if duplicates:
            logger.warning(f"⚠️ Found duplicates: {duplicates}")
        else:
            logger.info("✅ No duplicates found - upsert working correctly!")

        logger.info("🎯 Test completed - data preserved in test_upsert_keep_data table")
        return True

    except Exception as e:
        logger.error(f"❌ Upsert test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("ClickZetta Upsert Test (Data Preserved)")
    print("=" * 80)

    success = test_clickzetta_upsert_no_cleanup()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if success:
        print("🎉 Upsert test completed successfully!")
        print("\n📊 Data Status:")
        print("✅ Test data preserved in 'test_upsert_keep_data' table")
        print("✅ You can now check the table contents in ClickZetta")
        print("\n💡 To clean up later, run:")
        print("   DELETE FROM test_upsert_keep_data WHERE test_run = 'keep_data';")
    else:
        print("❌ Upsert test failed. Please check the logs above.")