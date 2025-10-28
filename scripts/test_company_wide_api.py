#!/usr/bin/env python3
"""
Company-wide modeling API test script
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/company-wide"

def test_modeling_status(org):
    """Test 1: Check modeling status"""
    print(f"\n{'='*60}")
    print(f"Test 1: Modeling Status ({org})")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/modeling/status", params={"organization": org})
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_setup(org):
    """Test 2: Setup PyCaret with augmentation"""
    print(f"\n{'='*60}")
    print(f"Test 2: Setup PyCaret + Augmentation ({org})")
    print(f"{'='*60}")

    data = {
        "organization": org,
        "use_augmentation": True,
        "target_size": 200
    }

    response = requests.post(f"{BASE_URL}/modeling/setup", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def test_compare(org):
    """Test 3: Compare models"""
    print(f"\n{'='*60}")
    print(f"Test 3: Compare Models ({org})")
    print(f"{'='*60}")

    data = {
        "organization": org,
        "n_select": 3
    }

    response = requests.post(f"{BASE_URL}/modeling/compare", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()

    # 비교 데이터가 크므로 요약만 출력
    if 'comparison_data' in result and result['comparison_data']:
        print(f"Models compared: {len(result['comparison_data'])}")
        print(f"Best model: {result.get('recommended_model_type')}")
        print(f"\nTop 3 models by R2:")
        for i, model in enumerate(result['comparison_data'][:3]):
            print(f"  {i+1}. {model.get('Model', 'Unknown')}: R2 = {model.get('R2', 'N/A')}")
    else:
        print(json.dumps(result, indent=2))

    return result

def test_train(org, model_name):
    """Test 4: Train specific model"""
    print(f"\n{'='*60}")
    print(f"Test 4: Train Model ({org}, {model_name})")
    print(f"{'='*60}")

    data = {
        "organization": org,
        "model_name": model_name
    }

    response = requests.post(f"{BASE_URL}/modeling/train", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def test_prediction(org):
    """Test 5: Get 2026 prediction"""
    print(f"\n{'='*60}")
    print(f"Test 5: 2026 Prediction ({org})")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/dashboard/prediction", params={"organization": org})
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def test_importance(org):
    """Test 6: Get feature importance"""
    print(f"\n{'='*60}")
    print(f"Test 6: Feature Importance ({org})")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/dashboard/importance", params={"organization": org, "top_n": 10})
    print(f"Status: {response.status_code}")
    result = response.json()

    if 'features' in result:
        print(f"\nTop {len(result['features'])} features:")
        for i, feat in enumerate(result['features']):
            print(f"  {i+1}. {feat['label']}: {feat['importance']:.4f}")
    else:
        print(json.dumps(result, indent=2))

    return result

def test_trend(org):
    """Test 7: Get trend data"""
    print(f"\n{'='*60}")
    print(f"Test 7: Trend Data ({org})")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/dashboard/trend", params={"organization": org})
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result

def main():
    org = sys.argv[1] if len(sys.argv) > 1 else "R&A"

    print(f"\n{'#'*60}")
    print(f"# Company-Wide Modeling API Test: {org}")
    print(f"{'#'*60}")

    try:
        # Test 1: Status
        test_modeling_status(org)

        # Test 2: Setup + Augmentation
        setup_result = test_setup(org)

        if setup_result.get('data_info', {}).get('augmented_size'):
            print(f"\n✅ Augmentation successful: {setup_result['data_info']['augmented_size']} rows")

        # Test 3: Compare models
        compare_result = test_compare(org)

        # Test 4: Train best model
        if compare_result.get('comparison_data'):
            best_model = compare_result['comparison_data'][0].get('Model', 'lr')
            test_train(org, best_model)

        # Test 5: Prediction
        test_prediction(org)

        # Test 6: Feature importance
        test_importance(org)

        # Test 7: Trend
        test_trend(org)

        print(f"\n{'='*60}")
        print("✅ All tests completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
