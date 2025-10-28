# Electron 변환 작업 계획

## 📋 프로젝트 개요

**목표**: Hwaseung Dashboard를 Electron 데스크톱 앱으로 변환
**참조 프로젝트**: `C:\Project\2025_wage_prediction` (SambioWage)
**예상 소요 시간**: 4-6시간

---

## ✅ 사전 준비 완료 항목

- [x] 2025_wage_prediction 프로젝트 구조 분석
- [x] hwaseung 프로젝트 아키텍처 파악
- [x] DB 파일 최적화 (95MB → 0.21MB)
- [x] 상세 계획 수립

---

## 📦 Phase 1: 프로젝트 구조 설정

### 1.1 필수 패키지 설치
```bash
cd C:\Project\hwaseung\frontend
npm install --save express
npm install --save-dev electron electron-builder typescript
npm install --save-dev @types/electron @types/express @types/node
npm install --save-dev cross-env
```

- [ ] 패키지 설치 완료
- [ ] package.json 확인

### 1.2 디렉토리 구조 생성
```
hwaseung/
├── electron/
│   ├── main.ts
│   ├── preload.ts
│   ├── tsconfig.json
│   └── icon.ico
├── dist-electron/          # 컴파일 결과
├── resources/              # Python Runtime
│   └── python-runtime/
└── release/                # 빌드 결과
```

- [ ] `electron/` 폴더 생성
- [ ] `resources/python-runtime/` 폴더 생성
- [ ] `dist-electron/` 폴더 생성 (자동)
- [ ] `release/` 폴더 생성 (자동)

---

## 📝 Phase 2: package.json 수정

### 2.1 main 필드 추가
```json
{
  "main": "dist-electron/main.js"
}
```

- [ ] `main` 필드 추가

### 2.2 scripts 추가
```json
{
  "scripts": {
    "electron:compile": "tsc -p electron/tsconfig.json",
    "electron:dev": "npm run electron:compile && cross-env NODE_ENV=development electron .",
    "electron:build:win": "npm run build && npm run electron:compile && electron-builder --win --x64",
    "electron:build:mac": "npm run build && npm run electron:compile && electron-builder --mac"
  }
}
```

- [ ] scripts 추가 완료

### 2.3 electron-builder 설정 추가
```json
{
  "build": {
    "appId": "com.hwaseung.dashboard",
    "productName": "Hwaseung Dashboard",
    "asar": false,
    "directories": {
      "output": "release"
    },
    "files": [
      "dist-electron/**/*",
      "build/**/*",
      "../backend/app/**/*",
      "../backend/data/**/*",
      "../backend/models/**/*",
      "../backend/run.py",
      "../backend/db_config.py",
      "../backend/requirements.txt",
      "../resources/python-runtime/**/*",
      "node_modules/express/**/*",
      "!../backend/venv/**/*",
      "!../backend/__pycache__",
      "!**/*.pyc"
    ],
    "extraResources": [
      {
        "from": "../resources/python-runtime",
        "to": "python-runtime"
      },
      {
        "from": "../backend/data",
        "to": "data"
      },
      {
        "from": "../backend/models",
        "to": "models"
      },
      {
        "from": "../hwaseung_RnD.db",
        "to": "hwaseung_RnD.db"
      }
    ],
    "win": {
      "target": [
        {
          "target": "portable",
          "arch": ["x64"]
        }
      ],
      "signAndEditExecutable": false
    },
    "mac": {
      "target": ["dmg"],
      "category": "public.app-category.business"
    }
  }
}
```

- [ ] electron-builder 설정 추가 완료

---

## 🔧 Phase 3: TypeScript 파일 생성

### 3.1 electron/tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "../dist-electron",
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "moduleResolution": "node",
    "types": ["node", "electron"]
  },
  "include": ["*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] `electron/tsconfig.json` 생성

### 3.2 electron/preload.ts
```typescript
/**
 * Preload script for Hwaseung Dashboard Electron App
 */

console.log('Preload script loaded');

export {};
```

- [ ] `electron/preload.ts` 생성

### 3.3 electron/main.ts
**핵심 기능:**
1. Python 백엔드 프로세스 시작 (port 8000)
2. Express 프론트엔드 서버 시작 (port 3000)
3. DB 파일 관리 (사용자 폴더에 복사)
4. 로깅 시스템
5. 프로세스 정리

**참조**: `C:\Project\2025_wage_prediction\electron\main.ts`

**주요 수정 사항:**
- `setupDatabase()` 함수 추가 (DB 복사 로직)
- 환경변수에 `DB_PATH` 추가
- 백엔드 경로: `backend/run.py`

- [ ] `electron/main.ts` 생성
- [ ] `setupDatabase()` 함수 구현
- [ ] Python 프로세스 시작 로직 구현
- [ ] Express 서버 시작 로직 구현
- [ ] 윈도우 생성 로직 구현
- [ ] 정리(cleanup) 함수 구현

---

## 🐍 Phase 4: Backend 수정

### 4.1 backend/run.py 수정
Electron 환경 감지 추가:
```python
import os
import sys

# Electron 앱에서 실행 중인지 확인
is_electron = os.getenv('ELECTRON_APP') == 'true'

if __name__ == "__main__":
    import uvicorn

    # Electron 환경에서는 콘솔 출력 강제
    if is_electron:
        sys.stdout = sys.stderr
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print("Starting server on 127.0.0.1:8000")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
```

- [ ] `backend/run.py` 수정 완료
- [ ] Electron 환경 감지 코드 추가
- [ ] 인코딩 설정 추가

### 4.2 backend/db_config.py 확인
현재 `DB_PATH` 환경변수를 이미 지원하므로 수정 불필요

- [x] db_config.py 확인 (수정 불필요)

---

## 🐍 Phase 5: Python Runtime 준비

### 5.1 Python Embedded 다운로드
- **버전**: Python 3.10.x Embedded
- **링크**: https://www.python.org/downloads/windows/
- **파일**: `python-3.10.x-embed-amd64.zip`

- [ ] Python Embedded 다운로드
- [ ] `resources/python-runtime/` 폴더에 압축 해제

### 5.2 pip 설치
```bash
# python-runtime 폴더에서
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python.exe get-pip.py
```

- [ ] pip 설치 완료

### 5.3 의존성 패키지 설치
```bash
cd resources/python-runtime
python.exe -m pip install -r ../../backend/requirements.txt --target ./Lib/site-packages
```

- [ ] FastAPI 설치
- [ ] PyCaret 설치
- [ ] 기타 의존성 설치
- [ ] 설치 완료 확인

### 5.4 python310._pth 수정
```
python310.zip
.
./Lib/site-packages
import site
```

- [ ] `python310._pth` 파일 수정

---

## 🧪 Phase 6: 개발 모드 테스트

### 6.1 Frontend 빌드
```bash
cd frontend
npm run build
```

- [ ] Frontend 빌드 완료
- [ ] `build/` 폴더 확인

### 6.2 Electron TypeScript 컴파일
```bash
npm run electron:compile
```

- [ ] TypeScript 컴파일 완료
- [ ] `dist-electron/main.js` 생성 확인
- [ ] `dist-electron/preload.js` 생성 확인

### 6.3 개발 모드 실행
```bash
npm run electron:dev
```

**확인 사항:**
- [ ] Electron 윈도우가 열리는가?
- [ ] Backend 프로세스가 시작되는가?
- [ ] Frontend가 로드되는가?
- [ ] API 통신이 정상인가?
- [ ] DB 파일이 복사되는가?
- [ ] 모든 페이지가 작동하는가?

### 6.4 로그 확인
- [ ] `electron-main.log` 확인
- [ ] `backend.log` 확인
- [ ] 에러 없음 확인

---

## 📦 Phase 7: 프로덕션 빌드

### 7.1 Windows 빌드
```bash
cd frontend
npm run electron:build:win
```

**빌드 시간**: 약 10-20분 (Python 패키징)

- [ ] 빌드 시작
- [ ] 빌드 완료 (에러 없음)
- [ ] `release/` 폴더 확인

### 7.2 빌드 결과 확인
- [ ] `Hwaseung Dashboard.exe` 파일 존재
- [ ] 파일 크기 확인 (~200-300MB)
- [ ] 실행 가능 여부 확인

### 7.3 실행 테스트
```bash
cd release
./Hwaseung Dashboard.exe
```

**확인 사항:**
- [ ] 앱이 실행되는가?
- [ ] Python 서버가 시작되는가?
- [ ] Frontend가 로드되는가?
- [ ] 모든 기능이 작동하는가?
- [ ] 로그 파일 생성 확인

---

## 🔍 Phase 8: 최종 검증

### 8.1 기능 테스트
- [ ] 데이터 업로드 페이지
- [ ] 전사 모델링 페이지
- [ ] 전사 분석 페이지
- [ ] 조직 인력 검토 페이지
- [ ] 대시보드 (R&A)
- [ ] 대시보드 (통기본)
- [ ] 모든 API 엔드포인트

### 8.2 성능 테스트
- [ ] 앱 시작 시간 (<10초)
- [ ] 페이지 로딩 속도
- [ ] 메모리 사용량
- [ ] CPU 사용량

### 8.3 에러 처리 테스트
- [ ] Backend 크래시 시 에러 메시지
- [ ] Frontend 로드 실패 시
- [ ] DB 파일 없을 때
- [ ] 포트 충돌 시

---

## 📚 Phase 9: 문서화

### 9.1 README 업데이트
- [ ] Electron 빌드 방법 추가
- [ ] 실행 방법 추가
- [ ] 트러블슈팅 가이드 추가

### 9.2 사용자 매뉴얼
- [ ] 설치 방법
- [ ] 실행 방법
- [ ] 주의사항
- [ ] FAQ

---

## 🎯 Phase 10: 배포 준비

### 10.1 배포 파일 정리
- [ ] 불필요한 파일 제거
- [ ] 최종 빌드 생성
- [ ] 압축 파일 생성

### 10.2 테스트 환경에서 검증
- [ ] 깨끗한 Windows PC에서 테스트
- [ ] 다른 사용자 계정에서 테스트
- [ ] 네트워크 환경 테스트

---

## 🔧 예상 문제 및 해결방안

### 문제 1: Python 패키지 누락
**증상**: Backend 시작 실패, ModuleNotFoundError
**해결**: resources/python-runtime/Lib/site-packages에 패키지 재설치

### 문제 2: DB 파일 접근 권한
**증상**: Database file not found
**해결**: DB 경로 확인, 파일 권한 확인

### 문제 3: 포트 충돌
**증상**: Backend/Frontend 서버 시작 실패
**해결**: 사용 중인 프로세스 종료 또는 포트 변경

### 문제 4: 인코딩 에러
**증상**: 한글 깨짐, UnicodeEncodeError
**해결**: run.py에서 UTF-8 인코딩 설정 확인

---

## 📊 예상 결과물

### 최종 파일
- **실행 파일**: `Hwaseung Dashboard.exe`
- **크기**: ~200-300MB
- **형태**: Portable (설치 불필요)
- **지원 OS**: Windows 10/11 (x64)

### 포함 내용
- React Frontend (빌드됨)
- FastAPI Backend
- Python 3.10 Runtime
- ML 모델 파일
- hwaseung_RnD.db (0.21MB)

---

## ⏱️ 예상 소요 시간

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1-2 | 패키지 설치 및 설정 | 30분 |
| 3 | TypeScript 파일 작성 | 1시간 |
| 4 | Backend 수정 | 30분 |
| 5 | Python Runtime 준비 | 1시간 |
| 6 | 개발 모드 테스트 | 1시간 |
| 7 | 프로덕션 빌드 | 30분 |
| 8 | 최종 검증 | 1시간 |
| 9-10 | 문서화 및 배포 | 30분 |
| **총계** | | **6-7시간** |

---

## 🚀 시작 준비 완료

현재 상태:
- ✅ DB 최적화 완료 (0.21MB)
- ✅ 참조 프로젝트 분석 완료
- ✅ 상세 계획 수립 완료

다음 작업:
1. Phase 1-2: 패키지 설치 및 package.json 수정
2. Phase 3: TypeScript 파일 생성

**준비되면 Phase 1부터 시작하세요!**

---

## 📝 체크리스트 요약

- [ ] Phase 1: 프로젝트 구조 설정
- [ ] Phase 2: package.json 수정
- [ ] Phase 3: TypeScript 파일 생성
- [ ] Phase 4: Backend 수정
- [ ] Phase 5: Python Runtime 준비
- [ ] Phase 6: 개발 모드 테스트
- [ ] Phase 7: 프로덕션 빌드
- [ ] Phase 8: 최종 검증
- [ ] Phase 9: 문서화
- [ ] Phase 10: 배포 준비

**현재 진행률: 0/10 완료**
