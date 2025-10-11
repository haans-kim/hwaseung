import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import {
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Database,
  PlayCircle,
  BarChart3,
  Target,
  Zap,
  Users
} from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

interface OrganizationStatus {
  organization: string;
  has_data: boolean;
  data_rows: number;
  environment_setup: boolean;
  model_trained: boolean;
  models_compared: boolean;
  current_model_type: string | null;
  augmented_size?: number;
  best_model?: string;
  best_r2?: number;
}

interface SetupResult {
  organization: string;
  original_size: number;
  augmented_size: number;
  feature_count: number;
}

interface ComparisonResult {
  organization: string;
  models_compared: number;
  best_model: string;
  comparison_data: Array<{
    Model: string;
    MAE: number;
    MSE: number;
    RMSE: number;
    R2: number;
  }>;
}

interface TrainingResult {
  organization: string;
  model_type: string;
  model_name: string;
  metrics: {
    R2: number;
    MAE: number;
    RMSE: number;
  };
}

const CompanyWideModeling: React.FC = () => {
  // 각 조직별 상태
  const [rnaStatus, setRnaStatus] = useState<OrganizationStatus | null>(null);
  const [tongStatus, setTongStatus] = useState<OrganizationStatus | null>(null);

  // 각 조직별 결과
  const [rnaSetup, setRnaSetup] = useState<SetupResult | null>(null);
  const [tongSetup, setTongSetup] = useState<SetupResult | null>(null);

  const [rnaComparison, setRnaComparison] = useState<ComparisonResult | null>(null);
  const [tongComparison, setTongComparison] = useState<ComparisonResult | null>(null);

  const [rnaTraining, setRnaTraining] = useState<TrainingResult | null>(null);
  const [tongTraining, setTongTraining] = useState<TrainingResult | null>(null);

  // 선택된 모델
  const [rnaSelectedModel, setRnaSelectedModel] = useState<string>('lr');
  const [tongSelectedModel, setTongSelectedModel] = useState<string>('lr');

  // 로딩 상태
  const [loading, setLoading] = useState<{[key: string]: boolean}>({});
  const [error, setError] = useState<string | null>(null);

  // 초기 상태 조회
  useEffect(() => {
    fetchAllStatus();
  }, []);

  const fetchAllStatus = async () => {
    try {
      const [rnaRes, tongRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/company-wide/modeling/status?organization=R%26A`),
        fetch(`${API_BASE_URL}/api/company-wide/modeling/status?organization=tonggibon`)
      ]);

      if (rnaRes.ok) {
        const rnaData = await rnaRes.json();
        setRnaStatus(rnaData);
      }

      if (tongRes.ok) {
        const tongData = await tongRes.json();
        setTongStatus(tongData);
      }
    } catch (err) {
      console.error('상태 조회 실패:', err);
    }
  };

  // 1단계: 데이터 증강 (양쪽 동시 진행)
  const handleDataAugmentation = async () => {
    try {
      setLoading({ ...loading, augmentation: true });
      setError(null);

      // R&A와 tonggibon 동시 Setup
      const [rnaRes, tongRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/company-wide/modeling/setup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization: 'R&A',
            use_augmentation: true,
            target_size: 200
          })
        }),
        fetch(`${API_BASE_URL}/api/company-wide/modeling/setup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization: 'tonggibon',
            use_augmentation: true,
            target_size: 200
          })
        })
      ]);

      if (!rnaRes.ok || !tongRes.ok) {
        throw new Error('데이터 증강 실패');
      }

      const rnaData = await rnaRes.json();
      const tongData = await tongRes.json();

      setRnaSetup({
        organization: 'R&A',
        original_size: rnaData.data_info.original_size,
        augmented_size: rnaData.data_info.augmented_size,
        feature_count: rnaData.data_info.feature_count
      });

      setTongSetup({
        organization: 'tonggibon',
        original_size: tongData.data_info.original_size,
        augmented_size: tongData.data_info.augmented_size,
        feature_count: tongData.data_info.feature_count
      });

      // 상태 갱신
      await fetchAllStatus();
    } catch (err: any) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading({ ...loading, augmentation: false });
    }
  };

  // 2단계: 모델 비교 (양쪽 동시 진행)
  const handleModelComparison = async () => {
    try {
      setLoading({ ...loading, comparison: true });
      setError(null);

      // R&A와 tonggibon 동시 Compare
      const [rnaRes, tongRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/company-wide/modeling/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization: 'R&A',
            n_select: 3
          })
        }),
        fetch(`${API_BASE_URL}/api/company-wide/modeling/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization: 'tonggibon',
            n_select: 3
          })
        })
      ]);

      if (!rnaRes.ok || !tongRes.ok) {
        throw new Error('모델 비교 실패');
      }

      const rnaData = await rnaRes.json();
      const tongData = await tongRes.json();

      setRnaComparison(rnaData);
      setTongComparison(tongData);

      // 최고 모델 자동 선택
      if (rnaData.comparison_data && rnaData.comparison_data.length > 0) {
        const modelMap: {[key: string]: string} = {
          'Linear Regression': 'lr',
          'Ridge Regression': 'ridge',
          'Lasso Regression': 'lasso',
          'Random Forest Regressor': 'rf',
          'Gradient Boosting Regressor': 'gbr'
        };
        setRnaSelectedModel(modelMap[rnaData.comparison_data[0].Model] || 'lr');
      }

      if (tongData.comparison_data && tongData.comparison_data.length > 0) {
        const modelMap: {[key: string]: string} = {
          'Linear Regression': 'lr',
          'Ridge Regression': 'ridge',
          'Lasso Regression': 'lasso',
          'Random Forest Regressor': 'rf',
          'Gradient Boosting Regressor': 'gbr'
        };
        setTongSelectedModel(modelMap[tongData.comparison_data[0].Model] || 'lr');
      }

      await fetchAllStatus();
    } catch (err: any) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading({ ...loading, comparison: false });
    }
  };

  // 3단계: R&A 모델 학습
  const handleRnaTrain = async () => {
    try {
      setLoading({ ...loading, rnaTrain: true });
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/company-wide/modeling/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          organization: 'R&A',
          model_name: rnaSelectedModel
        })
      });

      if (!response.ok) {
        throw new Error('R&A 모델 학습 실패');
      }

      const data = await response.json();
      setRnaTraining(data);

      await fetchAllStatus();
    } catch (err: any) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading({ ...loading, rnaTrain: false });
    }
  };

  // 3단계: 통합기술본부 모델 학습
  const handleTongTrain = async () => {
    try {
      setLoading({ ...loading, tongTrain: true });
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/company-wide/modeling/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          organization: 'tonggibon',
          model_name: tongSelectedModel
        })
      });

      if (!response.ok) {
        throw new Error('통합기술본부 모델 학습 실패');
      }

      const data = await response.json();
      setTongTraining(data);

      await fetchAllStatus();
    } catch (err: any) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading({ ...loading, tongTrain: false });
    }
  };

  const modelNameKorean: {[key: string]: string} = {
    'lr': '선형 회귀',
    'ridge': 'Ridge 회귀',
    'lasso': 'Lasso 회귀',
    'en': 'Elastic Net',
    'rf': 'Random Forest',
    'gbr': 'Gradient Boosting'
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 헤더 */}
      <div>
        <h1 className="text-3xl font-bold">전사 적정인력 산정 모델링</h1>
        <p className="text-muted-foreground">R&A와 통합기술본부의 2026년 적정인력 예측 모델 학습</p>
      </div>

      {/* 에러 */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 1단계: 데이터 증강 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            1단계: 데이터 증강
          </CardTitle>
          <CardDescription>
            R&A와 통합기술본부의 원본 데이터를 각각 200개로 증강합니다.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* R&A 현황 */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">R&A</h4>
                {rnaSetup && <CheckCircle2 className="h-5 w-5 text-green-500" />}
              </div>
              {rnaStatus && (
                <div className="space-y-1 text-sm">
                  <p>원본 데이터: {rnaStatus.data_rows}개</p>
                  {rnaSetup && <p className="text-green-600">증강 완료: {rnaSetup.augmented_size}개</p>}
                </div>
              )}
            </div>

            {/* 통합기술본부 현황 */}
            <div className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold">통합기술본부</h4>
                {tongSetup && <CheckCircle2 className="h-5 w-5 text-green-500" />}
              </div>
              {tongStatus && (
                <div className="space-y-1 text-sm">
                  <p>원본 데이터: {tongStatus.data_rows}개</p>
                  {tongSetup && <p className="text-green-600">증강 완료: {tongSetup.augmented_size}개</p>}
                </div>
              )}
            </div>
          </div>

          <Button
            onClick={handleDataAugmentation}
            disabled={loading.augmentation || !rnaStatus?.has_data || !tongStatus?.has_data}
            className="w-full"
          >
            {loading.augmentation ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                데이터 증강 중...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                데이터 증강 시작
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* 2단계: 모델 비교 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            2단계: 적정 모델 산정
          </CardTitle>
          <CardDescription>
            각 조직별로 여러 모델을 비교하여 최적 모델을 찾습니다.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button
            onClick={handleModelComparison}
            disabled={loading.comparison || !rnaSetup || !tongSetup}
            className="w-full"
          >
            {loading.comparison ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                모델 비교 중...
              </>
            ) : (
              <>
                <PlayCircle className="mr-2 h-4 w-4" />
                모델 비교 시작
              </>
            )}
          </Button>

          {/* 비교 결과 */}
          {(rnaComparison || tongComparison) && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              {/* R&A 결과 */}
              {rnaComparison && (
                <div className="border rounded-lg p-4">
                  <h4 className="font-semibold mb-3">R&A 모델 비교 결과</h4>
                  <div className="space-y-2">
                    {rnaComparison.comparison_data.slice(0, 3).map((model, idx) => (
                      <div key={idx} className="flex justify-between items-center text-sm p-2 bg-muted rounded">
                        <span className="font-medium">{model.Model}</span>
                        <span className="text-blue-600">R² {model.R2 != null ? model.R2.toFixed(3) : 'N/A'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 통합기술본부 결과 */}
              {tongComparison && (
                <div className="border rounded-lg p-4">
                  <h4 className="font-semibold mb-3">통합기술본부 모델 비교 결과</h4>
                  <div className="space-y-2">
                    {tongComparison.comparison_data.slice(0, 3).map((model, idx) => (
                      <div key={idx} className="flex justify-between items-center text-sm p-2 bg-muted rounded">
                        <span className="font-medium">{model.Model}</span>
                        <span className="text-blue-600">R² {model.R2 != null ? model.R2.toFixed(3) : 'N/A'}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 3단계: 모델 학습 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            3단계: 모델 학습
          </CardTitle>
          <CardDescription>
            선택된 최적 모델을 각 조직별로 학습합니다.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            {/* R&A 학습 */}
            <div className="border rounded-lg p-4 space-y-3">
              <h4 className="font-semibold flex items-center gap-2">
                <Users className="h-4 w-4" />
                R&A
              </h4>

              {rnaComparison && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">모델 선택:</label>
                  <select
                    value={rnaSelectedModel}
                    onChange={(e) => setRnaSelectedModel(e.target.value)}
                    className="w-full p-2 border rounded"
                  >
                    {rnaComparison.comparison_data.map((model) => {
                      const modelCode = {
                        'Linear Regression': 'lr',
                        'Ridge Regression': 'ridge',
                        'Lasso Regression': 'lasso',
                        'Random Forest Regressor': 'rf',
                        'Gradient Boosting Regressor': 'gbr'
                      }[model.Model] || 'lr';

                      return (
                        <option key={modelCode} value={modelCode}>
                          {model.Model} (R² {model.R2 != null ? model.R2.toFixed(3) : 'N/A'})
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              <Button
                onClick={handleRnaTrain}
                disabled={loading.rnaTrain || !rnaComparison}
                className="w-full"
              >
                {loading.rnaTrain ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    학습 중...
                  </>
                ) : (
                  <>
                    <Target className="mr-2 h-4 w-4" />
                    R&A 모델 학습
                  </>
                )}
              </Button>

              {rnaTraining && (
                <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200">
                  <CheckCircle2 className="h-5 w-5 text-green-600 mb-2" />
                  <div className="text-sm space-y-1">
                    <p className="font-medium">학습 완료: {modelNameKorean[rnaTraining.model_type]}</p>
                    <p>R²: {rnaTraining.metrics.R2 != null ? rnaTraining.metrics.R2.toFixed(3) : 'N/A'}</p>
                    <p>MAE: {rnaTraining.metrics.MAE != null ? rnaTraining.metrics.MAE.toFixed(2) : 'N/A'}</p>
                    <p>RMSE: {rnaTraining.metrics.RMSE != null ? rnaTraining.metrics.RMSE.toFixed(2) : 'N/A'}</p>
                  </div>
                </div>
              )}
            </div>

            {/* 통합기술본부 학습 */}
            <div className="border rounded-lg p-4 space-y-3">
              <h4 className="font-semibold flex items-center gap-2">
                <Users className="h-4 w-4" />
                통합기술본부
              </h4>

              {tongComparison && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">모델 선택:</label>
                  <select
                    value={tongSelectedModel}
                    onChange={(e) => setTongSelectedModel(e.target.value)}
                    className="w-full p-2 border rounded"
                  >
                    {tongComparison.comparison_data.map((model) => {
                      const modelCode = {
                        'Linear Regression': 'lr',
                        'Ridge Regression': 'ridge',
                        'Lasso Regression': 'lasso',
                        'Random Forest Regressor': 'rf',
                        'Gradient Boosting Regressor': 'gbr'
                      }[model.Model] || 'lr';

                      return (
                        <option key={modelCode} value={modelCode}>
                          {model.Model} (R² {model.R2 != null ? model.R2.toFixed(3) : 'N/A'})
                        </option>
                      );
                    })}
                  </select>
                </div>
              )}

              <Button
                onClick={handleTongTrain}
                disabled={loading.tongTrain || !tongComparison}
                className="w-full"
              >
                {loading.tongTrain ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    학습 중...
                  </>
                ) : (
                  <>
                    <Target className="mr-2 h-4 w-4" />
                    통합기술본부 모델 학습
                  </>
                )}
              </Button>

              {tongTraining && (
                <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200">
                  <CheckCircle2 className="h-5 w-5 text-green-600 mb-2" />
                  <div className="text-sm space-y-1">
                    <p className="font-medium">학습 완료: {modelNameKorean[tongTraining.model_type]}</p>
                    <p>R²: {tongTraining.metrics.R2 != null ? tongTraining.metrics.R2.toFixed(3) : 'N/A'}</p>
                    <p>MAE: {tongTraining.metrics.MAE != null ? tongTraining.metrics.MAE.toFixed(2) : 'N/A'}</p>
                    <p>RMSE: {tongTraining.metrics.RMSE != null ? tongTraining.metrics.RMSE.toFixed(2) : 'N/A'}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 완료 안내 */}
      {rnaTraining && tongTraining && (
        <Alert>
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <AlertTitle>모델링 완료!</AlertTitle>
          <AlertDescription>
            <p className="mb-2">R&A와 통합기술본부의 모델 학습이 완료되었습니다.</p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => window.location.href = '/dashboard/rna'}
              >
                R&A Dashboard 보기
              </Button>
              <Button
                variant="outline"
                onClick={() => window.location.href = '/dashboard/tonggibon'}
              >
                통합기술본부 Dashboard 보기
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default CompanyWideModeling;
