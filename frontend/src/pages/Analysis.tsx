import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import {
  BarChart3,
  Wand2,
  AlertTriangle,
  Loader2,
  CheckCircle,
  TrendingUp,
} from 'lucide-react';
import { apiClient } from '../lib/api';

interface ShapAnalysis {
  available: boolean;
  explainer_type?: string;
  n_features?: number;
  feature_importance: Array<{
    feature: string;
    feature_korean?: string;
    importance: number;
  }>;
  error?: string;
}

interface FeatureImportance {
  feature: string;
  feature_korean?: string;
  importance: number;
  std?: number;
}

interface ModelPerformance {
  mae?: number;
  rmse?: number;
  r2?: number;
  mape?: number;
  residuals?: {
    mean: number;
    std: number;
  };
}

export default function Analysis() {
  const [shapAnalysis, setShapAnalysis] = useState<ShapAnalysis | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<number>(0);

  useEffect(() => {
    loadAnalysisData();
  }, []);

  const loadAnalysisData = async () => {
    setLoading('initial');
    setError(null);

    try {
      const [shapRes, featureRes, performanceRes] = await Promise.all([
        apiClient.getShapAnalysis(undefined, 20).catch(() => ({ available: false, error: 'SHAP 분석을 사용할 수 없습니다.' })),
        apiClient.getFeatureImportance('shap', 15).catch(() => 
          apiClient.getFeatureImportance('pycaret', 15).catch(() => ({ feature_importance: [], error: 'Feature importance 분석을 사용할 수 없습니다.' }))
        ),
        apiClient.getModelPerformance().catch(() => ({ performance: {}, error: '성능 분석을 사용할 수 없습니다.' }))
      ]);

      setShapAnalysis(shapRes);
      setFeatureImportance(featureRes.feature_importance || []);
      setModelPerformance(performanceRes);
    } catch (error) {
      setError('분석 데이터를 불러오는 중 오류가 발생했습니다.');
      console.error('Analysis data loading failed:', error);
    } finally {
      setLoading(null);
    }
  };

  const handleShapAnalysis = async () => {
    setLoading('shap');
    setError(null);

    try {
      const result = await apiClient.getShapAnalysis(selectedSample, 20);
      setShapAnalysis(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'SHAP 분석 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const formatNumber = (num: number, decimals: number = 4) => {
    return Number(num).toFixed(decimals);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Analysis</h1>
          <p className="text-muted-foreground">모델 해석 및 분석 결과</p>
        </div>
        <Button onClick={loadAnalysisData} disabled={loading === 'initial'}>
          {loading === 'initial' ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              로딩 중...
            </>
          ) : (
            '새로고침'
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 모델 성능 요약 */}
      {modelPerformance && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              모델 성능 지표
            </CardTitle>
            <CardDescription>
              학습된 모델의 예측 정확도
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {modelPerformance.mae !== undefined && (
                <div>
                  <p className="text-sm text-muted-foreground">MAE</p>
                  <p className="text-xl font-semibold">{formatNumber(modelPerformance.mae, 4)}</p>
                </div>
              )}
              {modelPerformance.rmse !== undefined && (
                <div>
                  <p className="text-sm text-muted-foreground">RMSE</p>
                  <p className="text-xl font-semibold">{formatNumber(modelPerformance.rmse, 4)}</p>
                </div>
              )}
              {modelPerformance.r2 !== undefined && (
                <div>
                  <p className="text-sm text-muted-foreground">R²</p>
                  <p className="text-xl font-semibold">{formatNumber(modelPerformance.r2, 4)}</p>
                </div>
              )}
              {modelPerformance.mape !== undefined && (
                <div>
                  <p className="text-sm text-muted-foreground">MAPE</p>
                  <p className="text-xl font-semibold">{formatNumber(modelPerformance.mape, 2)}%</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 해석 가능성 분석 - SHAP과 Feature Importance만 */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* SHAP 분석 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Wand2 className="mr-2 h-5 w-5" />
              SHAP 분석
              {shapAnalysis?.available && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              SHAP 값 기반 - 각 변수의 예측 기여도 분석
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">샘플 인덱스:</label>
              <input
                type="number"
                value={selectedSample}
                onChange={(e) => setSelectedSample(Number(e.target.value))}
                className="w-20 px-2 py-1 border border-border rounded-md"
                min="0"
              />
              <Button size="sm" onClick={handleShapAnalysis} disabled={loading === 'shap'}>
                {loading === 'shap' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  '분석 시작'
                )}
              </Button>
            </div>

            {shapAnalysis?.error ? (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{shapAnalysis.error}</AlertDescription>
              </Alert>
            ) : shapAnalysis?.available ? (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  {shapAnalysis.explainer_type} • {shapAnalysis.n_features}개 특성
                </p>
                {shapAnalysis.feature_importance.length > 0 && (
                  <div className="space-y-1">
                    {shapAnalysis.feature_importance.slice(0, 10).map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center py-1">
                        <span className="text-sm truncate mr-2">
                          {item.feature_korean || item.feature}
                        </span>
                        <span className="text-sm font-medium">{formatNumber(item.importance, 4)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                모델을 먼저 학습해주세요
              </div>
            )}
          </CardContent>
        </Card>

        {/* Feature 중요도 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="mr-2 h-5 w-5" />
              Feature 중요도
            </CardTitle>
            <CardDescription>
              변수별 예측 영향력 순위
            </CardDescription>
          </CardHeader>
          <CardContent>
            {featureImportance.length > 0 ? (
              <div className="space-y-1">
                {featureImportance.slice(0, 10).map((item, idx) => (
                  <div key={idx} className="flex justify-between items-center py-1">
                    <span className="text-sm truncate mr-2">
                      {item.feature_korean || item.feature}
                    </span>
                    <span className="text-sm font-medium">
                      {formatNumber(item.importance, 4)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                Feature importance 데이터가 없습니다
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}