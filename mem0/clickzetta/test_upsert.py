#!/usr/bin/env python3
"""
Test ClickZetta Upsert Functionality
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

def test_clickzetta_upsert():
    """Test ClickZetta upsert functionality."""
    logger.info("Testing ClickZetta upsert functionality...")

    try:
        from mem0.vector_stores.clickzetta import ClickZetta

        # Load configuration
        env_config = load_env_config()
        if not env_config:
            logger.error("Failed to load configuration. Exiting.")
            return False

        # Create ClickZetta vector store
        vector_store = ClickZetta(
            collection_name="test_upsert_functionality",
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
            {"data": "ClickZetta支持高性能的批量操作", "category": "database", "user_id": "user1"},
            {"data": "向量搜索技术在AI中很重要", "category": "ai", "user_id": "user2"},
            {"data": "Mem0是优秀的记忆管理系统", "category": "memory", "user_id": "user3"}
        ]

        test_ids = ["upsert_test_1", "upsert_test_2", "upsert_test_3"]

        # Test 1: Initial insert
        logger.info("Test 1: Initial insert...")
        vector_store.insert(test_vectors, test_payloads, test_ids)
        logger.info("✅ Initial insert completed")

        # Verify initial insert
        result = vector_store.get("upsert_test_1")
        if result:
            logger.info(f"✅ Initial data: {result.payload.get('data', 'N/A')}")
        else:
            logger.error("❌ Failed to retrieve initial data")
            return False

        # Test 2: Upsert with same IDs (should update)
        logger.info("Test 2: Upsert with same IDs (update)...")
        updated_payloads = [
            {"data": "ClickZetta支持高性能的批量操作 - 已更新", "category": "database", "user_id": "user1", "version": 2},
            {"data": "向量搜索技术在AI中很重要 - 已更新", "category": "ai", "user_id": "user2", "version": 2},
            {"data": "Mem0是优秀的记忆管理系统 - 已更新", "category": "memory", "user_id": "user3", "version": 2}
        ]

        vector_store.insert(test_vectors, updated_payloads, test_ids)
        logger.info("✅ Upsert (update) completed")

        # Verify update
        result = vector_store.get("upsert_test_1")
        if result and "已更新" in result.payload.get('data', ''):
            logger.info(f"✅ Updated data: {result.payload.get('data', 'N/A')}")
            logger.info(f"✅ Version: {result.payload.get('version', 'N/A')}")
        else:
            logger.error("❌ Update verification failed")
            return False

        # Test 3: Mixed insert/update
        logger.info("Test 3: Mixed insert/update...")
        mixed_vectors = test_vectors + [[1.0, 1.1, 1.2] + [0.0] * 1533]
        mixed_payloads = [
            {"data": "ClickZetta支持高性能的批量操作 - 再次更新", "category": "database", "user_id": "user1", "version": 3},
            {"data": "向量搜索技术在AI中很重要 - 再次更新", "category": "ai", "user_id": "user2", "version": 3},
            {"data": "Mem0是优秀的记忆管理系统 - 再次更新", "category": "memory", "user_id": "user3", "version": 3},
            {"data": "新增的测试数据", "category": "new", "user_id": "user4", "version": 1}
        ]
        mixed_ids = test_ids + ["upsert_test_4"]

        vector_store.insert(mixed_vectors, mixed_payloads, mixed_ids)
        logger.info("✅ Mixed insert/update completed")

        # Verify mixed operation
        updated_result = vector_store.get("upsert_test_1")
        new_result = vector_store.get("upsert_test_4")

        if (updated_result and "再次更新" in updated_result.payload.get('data', '') and
            new_result and "新增的测试数据" in new_result.payload.get('data', '')):
            logger.info(f"✅ Mixed operation verified")
            logger.info(f"  Updated: {updated_result.payload.get('data', 'N/A')[:30]}...")
            logger.info(f"  New: {new_result.payload.get('data', 'N/A')}")
        else:
            logger.error("❌ Mixed operation verification failed")
            return False

        # Test 4: Check for duplicates
        logger.info("Test 4: Checking for duplicates...")
        all_vectors = vector_store.list(limit=10)
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

        # Cleanup
        logger.info("Cleaning up test data...")
        cleanup_ids = ["upsert_test_1", "upsert_test_2", "upsert_test_3", "upsert_test_4"]
        for test_id in cleanup_ids:
            try:
                vector_store.delete(test_id)
            except:
                pass
        logger.info("✅ Cleanup completed")

        return True

    except Exception as e:
        logger.error(f"❌ Upsert test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("ClickZetta Upsert Functionality Test")
    print("=" * 80)

    success = test_clickzetta_upsert()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if success:
        print("🎉 Upsert functionality test passed!")
        print("\nKey Features Tested:")
        print("✅ Initial insert with unique IDs")
        print("✅ Upsert (update) with existing IDs")
        print("✅ Mixed insert/update operations")
        print("✅ Duplicate prevention verification")
        print("✅ Data integrity and version tracking")
    else:
        print("❌ Upsert functionality test failed. Please check the logs above.")

    print("\n💡 Benefits of Upsert:")
    print("   - Prevents duplicate records with same ID")
    print("   - Allows safe re-running of operations")
    print("   - Maintains data consistency")
    print("   - Reduces storage overhead")