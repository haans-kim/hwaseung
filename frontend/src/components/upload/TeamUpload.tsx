import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, CheckCircle, FileSpreadsheet, X } from 'lucide-react';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { API_BASE_URL } from '../../lib/api';

interface UploadResult {
  success: boolean;
  message: string;
  validation?: {
    companies: string[];
    teams: string[];
    years: number[];
    months: number[];
    positions: string[];
    row_count: number;
    team_count: number;
    feature_count: number;
  };
  saved?: {
    count: number;
    errors?: any;
  };
  error?: string;
}

export const TeamUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [dragActive, setDragActive] = useState(false);

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

      const response = await fetch(`${API_BASE_URL}/api/team/upload`, {
        method: 'POST',
        body: formData,
      });

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
          📋 조직인력산정용 Feature 업로드
        </h3>
        <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
          <p>• Excel 파일은 2개의 시트를 포함해야 합니다</p>
          <p>• Sheet 1 (Feature Matching): 조직 계층 정보 (HQ, 본부, 담당, 실, 팀)</p>
          <p>• Sheet 2 (Master): HQ, 팀, 년, 월, 구분, F1-F9, 인력규모</p>
          <p>• 팀별, 월별 Feature 데이터를 업로드합니다</p>
        </div>
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
          Excel 파일을 드래그하거나 클릭하여 선택하세요
        </p>
        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={handleFileChange}
          className="hidden"
          id="team-file-upload"
        />
        <label htmlFor="team-file-upload">
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
            {result.validation && (
              <div className="mt-2 text-sm space-y-1">
                <p>• 회사: {result.validation.companies?.join(', ') || 'N/A'}</p>
                <p>• 팀 수: {result.validation.team_count}개</p>
                <p>• Feature 수: {result.validation.feature_count}개</p>
                <p>• 데이터 행: {result.validation.row_count}개</p>
                <p>• 년도: {result.validation.years?.join(', ') || 'N/A'}</p>
                <p>• 월: {result.validation.months?.join(', ') || 'N/A'}</p>
                <p>• 직급: {result.validation.positions?.join(', ') || 'N/A'}</p>
              </div>
            )}
            {result.saved && (
              <div className="mt-2 text-sm space-y-1">
                <p>✅ {result.saved.count}개 데이터 저장 완료</p>
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

      {/* 현재 상태 */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <h4 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">
          💡 템플릿 형식
        </h4>
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
          <div>
            <p className="font-medium">Sheet 1: Feature Matching</p>
            <p className="ml-4">HQ | 본부 | 담당 | 실 | 팀</p>
          </div>
          <div>
            <p className="font-medium">Sheet 2: Master Data</p>
            <p className="ml-4">HQ | 팀 | 년 | 월 | 구분 | F1~F9 | 인력규모</p>
          </div>
        </div>
      </div>
    </div>
  );
};
