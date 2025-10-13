import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../ui/card';
import { Button } from '../ui/button';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';
import { Upload, FileSpreadsheet, CheckCircle, XCircle, AlertTriangle, Loader2, Database } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import { API_BASE_URL } from '../../lib/api';

interface CompanyWideUploadProps {
  organization: 'R&A' | 'tonggibon';
  title: string;
  description: string;
}

interface ValidationResult {
  rows: number;
  years: number[];
  warnings: string[];
}

interface SaveResult {
  saved_count: number;
  updated_count: number;
  total: number;
}

interface StoredDataRow {
  id?: number;
  organization: string;
  year: number;
  ev_growth_gl?: number;          // 글로벌 EV시장성장률
  v_growth_gl?: number;           // 글로벌 자동차 시장성장률
  v_export_kr?: number;           // 국내 자동차 수출액 증가율
  vp_export_kr?: number;          // 국내 자동차부품 수출액 증가율
  gdp_growth_kr?: number;         // GDP성장률
  cpi_kr?: number;                // 소비자물가상승률
  exchange_rate_change_krw?: number; // 환율변화율_원화기준
  scm_index_gl?: number;          // 글로벌물류비지수
  oil_gl?: number;                // 국제유가
  labor_cost?: number;            // 인건비 증감률
  revenue?: number;               // 매출액 증감률/증가율
  profit?: number;                // 영업이익 증감률/증가율
  operating_rate?: number;        // 가동률 증감률 or 연구개발비용 증감률
  operating_date?: number;        // 가동일수 증감률 or 연구개발정부보조금 증감률
  headcount?: number;             // 정원
  [key: string]: any;
}

export const CompanyWideUpload: React.FC<CompanyWideUploadProps> = ({
  organization,
  title,
  description
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [saveResult, setSaveResult] = useState<SaveResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [storedData, setStoredData] = useState<StoredDataRow[]>([]);
  const [loadingData, setLoadingData] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.xlsx')) {
        setFile(droppedFile);
        setError(null);
        setValidationResult(null);
        setSaveResult(null);
        setSuccess(null);
      } else {
        setError('Excel 파일(.xlsx)만 업로드 가능합니다.');
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.xlsx')) {
        setFile(selectedFile);
        setError(null);
        setValidationResult(null);
        setSaveResult(null);
        setSuccess(null);
      } else {
        setError('Excel 파일(.xlsx)만 업로드 가능합니다.');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('파일을 선택해주세요.');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);
    setValidationResult(null);
    setSaveResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(
        `${API_BASE_URL}/api/company-wide/upload?organization=${encodeURIComponent(organization)}`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '업로드 실패');
      }

      const result = await response.json();

      setValidationResult(result.validation);
      setSaveResult(result.save_result);
      setSuccess(`${organization} 데이터가 성공적으로 업로드되었습니다!`);

    } catch (err: any) {
      setError(err.message || '업로드 중 오류가 발생했습니다.');
    } finally {
      setUploading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setValidationResult(null);
    setSaveResult(null);
    setError(null);
    setSuccess(null);
  };

  // 저장된 데이터 불러오기
  const fetchStoredData = useCallback(async () => {
    setLoadingData(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/company-wide/features?organization=${encodeURIComponent(organization)}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }

      const result = await response.json();
      setStoredData(result.data || []);
    } catch (err) {
      console.error('Error fetching stored data:', err);
    } finally {
      setLoadingData(false);
    }
  }, [organization]);

  // 컴포넌트 마운트 시 데이터 로드
  useEffect(() => {
    fetchStoredData();
  }, [fetchStoredData]);

  // 업로드 성공 후 데이터 다시 로드
  useEffect(() => {
    if (success) {
      fetchStoredData();
    }
  }, [success, fetchStoredData]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* 파일 업로드 영역 */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-primary bg-primary/5'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id={`file-upload-${organization}`}
              className="hidden"
              accept=".xlsx"
              onChange={handleFileChange}
            />
            <label
              htmlFor={`file-upload-${organization}`}
              className="cursor-pointer flex flex-col items-center space-y-4"
            >
              <Upload className="h-12 w-12 text-gray-400" />
              <div>
                <p className="text-sm font-medium text-gray-700">
                  파일을 드래그하거나 클릭하여 선택하세요
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Excel 파일 (.xlsx) 형식만 지원
                </p>
              </div>
            </label>
          </div>

          {/* 파일 정보 표시 */}
          {file && (
            <Alert>
              <FileSpreadsheet className="h-4 w-4" />
              <AlertTitle>선택된 파일</AlertTitle>
              <AlertDescription>
                <div className="flex items-center justify-between">
                  <span className="text-sm">{file.name}</span>
                  <Button variant="ghost" size="sm" onClick={handleClear}>
                    취소
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* 파일 요구사항 */}
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>파일 요구사항</AlertTitle>
            <AlertDescription className="text-sm space-y-1">
              <p>• 형식: Excel (.xlsx)</p>
              <p>• 시트: master</p>
              <p>• 필수 컬럼: 16개 (외부환경 지표 10개, 내부지표 5개, 정원)</p>
              <p>• 데이터: 2021년부터 최신년도까지</p>
            </AlertDescription>
          </Alert>

          {/* 검증 결과 */}
          {validationResult && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertTitle>검증 완료</AlertTitle>
              <AlertDescription className="text-sm space-y-1">
                <p>• 데이터 행 수: {validationResult.rows}개</p>
                <p>• 데이터 기간: {validationResult.years.join(', ')}년</p>
                {validationResult.warnings.length > 0 && (
                  <div className="mt-2">
                    <p className="font-medium">경고:</p>
                    {validationResult.warnings.map((warning, idx) => (
                      <p key={idx} className="text-yellow-600">⚠️ {warning}</p>
                    ))}
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}

          {/* 저장 결과 */}
          {saveResult && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertTitle>저장 완료</AlertTitle>
              <AlertDescription className="text-sm space-y-1">
                <p>• 신규 저장: {saveResult.saved_count}개</p>
                <p>• 업데이트: {saveResult.updated_count}개</p>
                <p>• 총 {saveResult.total}개 데이터 처리 완료</p>
              </AlertDescription>
            </Alert>
          )}

          {/* 에러 메시지 */}
          {error && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertTitle>오류</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* 성공 메시지 */}
          {success && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertTitle>성공</AlertTitle>
              <AlertDescription>{success}</AlertDescription>
            </Alert>
          )}

          {/* 업로드 버튼 */}
          <div className="flex space-x-2">
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="flex-1"
            >
              {uploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  업로드 중...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  데이터베이스에 저장
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* 저장된 데이터 테이블 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            저장된 데이터
          </CardTitle>
          <CardDescription>
            데이터베이스에 저장된 {organization} 데이터 ({storedData.length}개 행)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadingData ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
              <span className="ml-2 text-gray-500">데이터 불러오는 중...</span>
            </div>
          ) : storedData.length === 0 ? (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>데이터 없음</AlertTitle>
              <AlertDescription>
                저장된 데이터가 없습니다. 위에서 파일을 업로드해주세요.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="rounded-md border overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="sticky left-0 bg-white z-10">연도</TableHead>
                    <TableHead>글로벌 EV시장성장률</TableHead>
                    <TableHead>글로벌 자동차 시장성장률</TableHead>
                    <TableHead>국내 자동차 수출액 증가율</TableHead>
                    <TableHead>국내 자동차부품 수출액 증가율</TableHead>
                    <TableHead>GDP성장률</TableHead>
                    <TableHead>소비자물가상승률</TableHead>
                    <TableHead>환율변화율(원화기준)</TableHead>
                    <TableHead>글로벌물류비지수</TableHead>
                    <TableHead>국제유가</TableHead>
                    <TableHead>인건비 증감률</TableHead>
                    <TableHead>{organization === 'R&A' ? '매출액 증감률' : '매출액 증가율'}</TableHead>
                    <TableHead>{organization === 'R&A' ? '영업이익 증감률' : '영업이익 증가율'}</TableHead>
                    <TableHead>{organization === 'R&A' ? '가동률 증감률' : '연구개발비용 증감률'}</TableHead>
                    <TableHead>{organization === 'R&A' ? '가동일수 증감률' : '연구개발정부보조금 증감률'}</TableHead>
                    <TableHead>정원</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {storedData.map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="sticky left-0 bg-white font-medium">{row.year}</TableCell>
                      <TableCell>{row.ev_growth_gl?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.v_growth_gl?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.v_export_kr?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.vp_export_kr?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.gdp_growth_kr?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.cpi_kr?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.exchange_rate_change_krw?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.scm_index_gl?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.oil_gl?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.labor_cost?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.revenue?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.profit?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.operating_rate?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.operating_date?.toFixed(2) ?? '-'}</TableCell>
                      <TableCell>{row.headcount ?? '-'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
