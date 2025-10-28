#!/usr/bin/env python3
"""
메모리 안전 모드로 모델 학습 테스트
실시간 메모리 모니터링
"""
import requests
import time
import subprocess
import threading
import json

BASE_URL = "http://localhost:8000"

def get_memory_usage():
    """Python 프로세스의 실제 메모리 사용량 조회"""
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )

    total_rss = 0
    max_rss = 0
    for line in result.stdout.split('\n'):
        if 'python' in line.lower() and 'run.py' in line:
            parts = line.split()
            if len(parts) >= 6:
                rss_kb = int(parts[5])
                total_rss += rss_kb
                max_rss = max(max_rss, rss_kb)

    return total_rss / 1024, max_rss / 1024  # MB 단위

class MemoryMonitor:
    def __init__(self):
        self.running = False
        self.memory_log = []
        self.peak_memory = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        while self.running:
            total, max_single = get_memory_usage()
            self.memory_log.append({
                'time': time.time(),
                'total_mb': total,
                'max_single_mb': max_single
            })
            self.peak_memory = max(self.peak_memory, total)
            time.sleep(1)  # 1초마다 측정

print("=" * 60)
print("메모리 안전 모드 모델 학습 테스트")
print("=" * 60)

# 초기 메모리 상태
initial_total, initial_max = get_memory_usage()
print(f"\n초기 메모리 사용량:")
print(f"  전체: {initial_total:.2f} MB")
print(f"  최대 단일 프로세스: {initial_max:.2f} MB")

# 메모리 모니터 시작
monitor = MemoryMonitor()
monitor.start()

try:
    print("\n" + "=" * 60)
    print("1단계: PyCaret 환경 설정")
    print("=" * 60)

    setup_data = {
        "target_column": "wage_increase_total_sbl",
        "session_id": 42,
        "train_size": 0.8
    }

    print(f"요청 데이터: {json.dumps(setup_data, indent=2, ensure_ascii=False)}")
    start_time = time.time()

    response = requests.post(f"{BASE_URL}/api/modeling/setup", json=setup_data)
    setup_time = time.time() - start_time

    current_total, current_max = get_memory_usage()
    print(f"\nSetup 완료 ({setup_time:.2f}초)")
    print(f"현재 메모리: {current_total:.2f} MB (+{current_total - initial_total:.2f} MB)")
    print(f"응답 상태: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Setup 성공")
        print(f"  - 데이터 크기: {result.get('data_info', {}).get('total_rows', 'N/A')} rows")
        print(f"  - Feature 수: {result.get('data_info', {}).get('feature_count', 'N/A')}")
    else:
        print(f"❌ Setup 실패: {response.text}")
        raise Exception("Setup failed")

    time.sleep(2)  # 메모리 안정화 대기

    print("\n" + "=" * 60)
    print("2단계: 모델 학습 (메모리 안전 모드)")
    print("=" * 60)
    print("🔍 실시간 메모리 모니터링 중...")

    start_time = time.time()

    response = requests.post(f"{BASE_URL}/api/modeling/compare")
    training_time = time.time() - start_time

    current_total, current_max = get_memory_usage()
    memory_increase = current_total - initial_total

    print(f"\n모델 학습 완료 ({training_time:.2f}초)")
    print(f"현재 메모리: {current_total:.2f} MB (+{memory_increase:.2f} MB)")
    print(f"피크 메모리: {monitor.peak_memory:.2f} MB")
    print(f"응답 상태: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ 모델 학습 성공!")
        print(f"  - 비교된 모델 수: {result.get('models_compared', 'N/A')}")
        print(f"  - 최고 모델 수: {result.get('best_model_count', 'N/A')}")
        print(f"  - 권장 모델: {result.get('recommended_model_type', 'N/A')}")
        print(f"  - 데이터 크기 카테고리: {result.get('data_size_category', 'N/A')}")
    else:
        print(f"❌ 학습 실패: {response.text}")

except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 모니터 중지
    monitor.stop()

    print("\n" + "=" * 60)
    print("메모리 사용 분석")
    print("=" * 60)

    final_total, final_max = get_memory_usage()

    print(f"\n초기 메모리:  {initial_total:.2f} MB")
    print(f"피크 메모리:  {monitor.peak_memory:.2f} MB")
    print(f"최종 메모리:  {final_total:.2f} MB")
    print(f"메모리 증가:  {final_total - initial_total:.2f} MB")

    # 메모리 추이 그래프 (간단한 ASCII)
    if monitor.memory_log:
        print("\n메모리 사용 추이 (MB):")
        print("-" * 60)

        min_mem = min(log['total_mb'] for log in monitor.memory_log)
        max_mem = max(log['total_mb'] for log in monitor.memory_log)
        mem_range = max_mem - min_mem if max_mem > min_mem else 1

        for i, log in enumerate(monitor.memory_log):
            if i % 5 == 0:  # 5초마다 출력
                mem = log['total_mb']
                bars = int((mem - min_mem) / mem_range * 40)
                bar_str = '█' * bars
                print(f"{i:3d}s: {bar_str} {mem:.1f} MB")

    print("\n" + "=" * 60)
    print("🎯 결과 평가")
    print("=" * 60)

    memory_increase_mb = monitor.peak_memory - initial_total

    if memory_increase_mb < 500:
        print(f"✅ 메모리 안전: 증가량 {memory_increase_mb:.2f} MB (< 500 MB)")
    elif memory_increase_mb < 1000:
        print(f"⚠️  주의 필요: 증가량 {memory_increase_mb:.2f} MB (500-1000 MB)")
    else:
        print(f"🚨 위험: 증가량 {memory_increase_mb:.2f} MB (> 1000 MB)")

    if final_total < initial_total + 200:
        print(f"✅ 메모리 정리 성공: GC가 제대로 작동함")
    else:
        print(f"⚠️  메모리 누수 가능성: {final_total - initial_total:.2f} MB 잔류")

    print("\n테스트 완료!")
