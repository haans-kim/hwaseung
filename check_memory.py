#!/usr/bin/env python3
"""
Activity Monitor의 메모리 수치 이해하기
"""
import subprocess
import sys

print("=" * 60)
print("Activity Monitor 메모리 수치 설명")
print("=" * 60)

# 현재 Python 프로세스 메모리 확인
result = subprocess.run(
    ['ps', '-o', 'pid,vsz,rss,comm', '-p', str(subprocess.os.getpid())],
    capture_output=True,
    text=True
)

lines = result.stdout.strip().split('\n')
if len(lines) > 1:
    header = lines[0]
    data = lines[1].split()

    pid = data[0]
    vsz_kb = int(data[1])
    rss_kb = int(data[2])

    print(f"\n현재 Python 프로세스 (PID: {pid}):")
    print(f"  VSZ (Virtual Size):  {vsz_kb / 1024 / 1024:.2f} GB")
    print(f"  RSS (실제 사용):     {rss_kb / 1024:.2f} MB")

print("\n" + "=" * 60)
print("Activity Monitor 용어 설명")
print("=" * 60)

explanations = """
1. 🟢 Memory (메모리) - 실제 물리 메모리 사용량
   - 실제로 RAM을 차지하는 양
   - 이 값이 시스템 전체 메모리를 초과하면 문제

2. 🟡 Virtual Memory (가상 메모리) - 할당된 주소 공간
   - 프로세스가 "사용할 수 있는" 주소 공간
   - 실제로 RAM을 쓰지 않음!
   - macOS는 기본적으로 매우 큰 값 (수백 GB)

3. 🔴 Compressed (압축됨)
   - macOS가 메모리 압축한 양
   - 실제 메모리 부족 시 압축해서 공간 절약

4. ⚪️ Swap Used (스왑 사용)
   - 디스크로 옮긴 메모리
   - 이 값이 크면 메모리 부족 신호

당신이 본 300GB는 아마도:
❌ Virtual Memory (가상 메모리) - 실제 사용량 아님!
✅ 실제 확인해야 할 값: Memory 또는 RSS
"""

print(explanations)

print("\n" + "=" * 60)
print("Python 프로세스 메모리 상세 분석")
print("=" * 60)

# 모든 Python 프로세스 찾기
result = subprocess.run(
    ['ps', 'aux'],
    capture_output=True,
    text=True
)

python_processes = [line for line in result.stdout.split('\n')
                   if 'python' in line.lower() and 'grep' not in line]

if python_processes:
    print("\n현재 실행 중인 Python 프로세스:")
    print(f"{'USER':<10} {'PID':<8} {'%MEM':<6} {'VSZ(GB)':<10} {'RSS(MB)':<10} {'COMMAND':<50}")
    print("-" * 100)

    total_rss = 0
    for line in python_processes[:10]:  # 최대 10개만
        parts = line.split()
        if len(parts) >= 11:
            user = parts[0]
            pid = parts[1]
            mem_pct = parts[3]
            vsz_kb = int(parts[4])
            rss_kb = int(parts[5])
            cmd = ' '.join(parts[10:])[:50]

            total_rss += rss_kb

            print(f"{user:<10} {pid:<8} {mem_pct:<6} {vsz_kb/1024/1024:<10.2f} {rss_kb/1024:<10.2f} {cmd:<50}")

    print("-" * 100)
    print(f"전체 Python 프로세스 실제 메모리 합계: {total_rss / 1024:.2f} MB ({total_rss / 1024 / 1024:.2f} GB)")

print("\n" + "=" * 60)
print("시스템 전체 메모리 현황")
print("=" * 60)

# 시스템 메모리 정보
result = subprocess.run(['vm_stat'], capture_output=True, text=True)
print(result.stdout[:500])

print("\n" + "=" * 60)
print("🎯 결론")
print("=" * 60)
print("""
300GB는 다음 중 하나입니다:

1. Virtual Memory (가상 메모리 주소 공간)
   → 실제 메모리 사용량이 아님!
   → macOS는 프로세스마다 수백 GB 할당 (정상)

2. 누적된 여러 프로세스의 Virtual Memory 합계
   → Activity Monitor에서 "All Processes" 선택 시 보이는 값
   → 이것도 실제 사용량 아님!

3. 시스템 전체 Swap 사용량
   → 디스크로 옮긴 메모리
   → 하지만 300GB는 너무 큼

✅ 확인 방법:
   Activity Monitor → 메모리 탭 → 하단의 "Physical Memory" 확인
   - Memory Used: 실제 사용 중인 메모리
   - Swap Used: 디스크로 옮긴 메모리
""")
