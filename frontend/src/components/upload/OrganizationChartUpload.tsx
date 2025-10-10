import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, CheckCircle, FileSpreadsheet, X } from 'lucide-react';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';

interface UploadResult {
  success: boolean;
  message: string;
  mode?: string;
  validation?: {
    row_count: number;
    companies: string[];
    company_count: number;
    division_count: number;
    department_count: number;
    office_count: number;
    team_count: number;
  };
  saved?: {
    deleted_count: number;
    saved_count: number;
    errors?: string[];
  };
  error?: string;
}

export const OrganizationChartUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [replaceAll, setReplaceAll] = useState(true);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.xlsx') || droppedFile.name.endsWith('.xls')) {
        setFile(droppedFile);
        setResult(null);
      } else {
        setResult({
          success: false,
          message: 'Excel 파일만 업로드 가능합니다 (.xlsx, .xls)',
        });
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(
        `http://localhost:8000/api/organization-chart/upload?replace_all=${replaceAll}`,
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setResult({
          success: false,
          message: data.detail || '업로드 실패',
          error: data.detail,
        });
      } else {
        setResult({
          success: true,
          message: data.message,
          mode: data.mode,
          validation: data.validation,
          saved: data.saved,
        });
        setFile(null);
      }
    } catch (error) {
      setResult({
        success: false,
        message: '서버 연결 실패',
        error: error instanceof Error ? error.message : '알 수 없는 오류',
      });
    } finally {
      setUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      {/* 안내 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
        <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
          🏢 조직도 데이터 업로드
        </h3>
        <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
          <p>• 필수 컬럼: 회사, 본부, 담당/사업단/센터, 실, 팀</p>
          <p>• 선택 컬럼: 비고</p>
          <p>• 계층 구조: 회사 → 본부 → 담당/사업단/센터 → 실 → 팀</p>
          <p>• 업로드 모드: 전체 교체 (기존 데이터 삭제 후 저장)</p>
        </div>
      </div>

      {/* 업로드 모드 선택 */}
      <div className="flex items-center space-x-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="radio"
            checked={replaceAll}
            onChange={() => setReplaceAll(true)}
            className="w-4 h-4"
          />
          <span className="text-sm">전체 교체 (기존 데이터 삭제)</span>
        </label>
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="radio"
            checked={!replaceAll}
            onChange={() => setReplaceAll(false)}
            className="w-4 h-4"
          />
          <span className="text-sm">추가/수정 (UPSERT)</span>
        </label>
      </div>

      {/* 드래그 앤 드롭 영역 */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${dragActive
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600'
          }
        `}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
          조직도 Excel 파일을 드래그하거나 클릭하여 선택하세요
        </p>
        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={handleFileChange}
          className="hidden"
          id="org-chart-file-upload"
        />
        <label htmlFor="org-chart-file-upload">
          <Button variant="outline" size="sm" className="cursor-pointer" asChild>
            <span>파일 선택</span>
          </Button>
        </label>
      </div>

      {/* 선택된 파일 */}
      {file && (
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="flex items-center space-x-3">
            <FileSpreadsheet className="h-8 w-8 text-green-600 dark:text-green-400" />
            <div>
              <p className="font-medium text-gray-900 dark:text-gray-100">{file.name}</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {(file.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              onClick={handleUpload}
              disabled={uploading}
              size="sm"
            >
              {uploading ? '업로드 중...' : '업로드'}
            </Button>
            <Button
              onClick={clearFile}
              variant="ghost"
              size="sm"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* 결과 표시 */}
      {result && (
        <Alert variant={result.success ? 'default' : 'destructive'}>
          {result.success ? (
            <CheckCircle className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <AlertDescription>
            <p className="font-semibold mb-2">{result.message}</p>
            {result.mode && (
              <p className="text-sm mb-2">모드: {result.mode === 'replace_all' ? '전체 교체' : 'UPSERT'}</p>
            )}
            {result.validation && (
              <div className="mt-2 text-sm space-y-1">
                <p>• 회사: {result.validation.companies.join(', ')}</p>
                <p>• 총 행 수: {result.validation.row_count}개</p>
                <p>• 본부 수: {result.validation.division_count}개</p>
                <p>• 담당/사업단/센터 수: {result.validation.department_count}개</p>
                <p>• 실 수: {result.validation.office_count}개</p>
                <p>• 팀 수: {result.validation.team_count}개</p>
              </div>
            )}
            {result.saved && (
              <div className="mt-2 text-sm space-y-1">
                {result.saved.deleted_count > 0 && (
                  <p>🗑️ 삭제된 데이터: {result.saved.deleted_count}개</p>
                )}
                <p>✅ 저장된 데이터: {result.saved.saved_count}개</p>
                {result.saved.errors && result.saved.errors.length > 0 && (
                  <div className="mt-2">
                    <p className="text-yellow-600 dark:text-yellow-400">⚠️ 경고:</p>
                    {result.saved.errors.slice(0, 5).map((error, index) => (
                      <p key={index} className="text-xs ml-4">{error}</p>
                    ))}
                    {result.saved.errors.length > 5 && (
                      <p className="text-xs ml-4">...외 {result.saved.errors.length - 5}개</p>
                    )}
                  </div>
                )}
              </div>
            )}
            {result.error && (
              <p className="mt-2 text-sm text-red-600 dark:text-red-400">
                오류: {result.error}
              </p>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* 데이터 형식 안내 */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">
          💡 데이터 형식
        </h4>
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
          <p className="font-medium">조직도 필수 컬럼:</p>
          <ul className="ml-4 space-y-1">
            <li>• 회사 (HQ)</li>
            <li>• 본부 (Division)</li>
            <li>• 담당/사업단/센터 (Department/Business Unit/Center)</li>
            <li>• 실 (Office)</li>
            <li>• 팀 (Team)</li>
            <li>• 비고 (Note) - 선택</li>
          </ul>
          <p className="mt-2 text-xs text-gray-500">
            * 전체 교체 모드: 기존 조직도 데이터를 모두 삭제하고 새 데이터로 교체합니다
          </p>
        </div>
      </div>
    </div>
  );
};
