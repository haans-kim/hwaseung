import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  BarChart3, 
  Brain, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Eye,
  Zap
} from 'lucide-react';
import { apiClient } from '../lib/api';

interface FeatureImportance {
  feature: string;
  importance: number;
  std?: number;
}

interface ShapAnalysis {
  available: boolean;
  feature_importance: FeatureImportance[];
  sample_explanation?: any;
  explainer_type: string;
  n_features: number;
  error?: string;
}

interface ModelPerformance {
  performance: {
    train_metrics: {
      mse: number;
      mae: number;
      r2: number;
    };
    test_metrics?: {
      mse: number;
      mae: number;
      r2: number;
    };
    residual_analysis: {
      mean_residual: number;
      std_residual: number;
      residual_range: [number, number];
    };
  };
  model_type: string;
  error?: string;
}

export const Analysis: React.FC = () => {
  const [shapAnalysis, setShapAnalysis] = useState<ShapAnalysis | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null);
  const [limeAnalysis, setLimeAnalysis] = useState<any>(null);
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
        apiClient.getFeatureImportance().catch(() => ({ feature_importance: [], error: 'Feature importance 분석을 사용할 수 없습니다.' })),
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

  const handleLimeAnalysis = async () => {
    setLoading('lime');
    setError(null);

    try {
      const result = await apiClient.getLimeAnalysis(selectedSample);
      setLimeAnalysis(result);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'LIME 분석 중 오류가 발생했습니다.');
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
      {modelPerformance && !modelPerformance.error && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              모델 성능 요약
            </CardTitle>
            <CardDescription>
              {modelPerformance.model_type} 모델의 성능 메트릭
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-background border p-4 rounded-lg">
                <h4 className="font-medium text-foreground">R² Score</h4>
                <p className="text-2xl font-bold text-primary">
                  {formatNumber(modelPerformance.performance.train_metrics?.r2 || 0, 3)}
                </p>
                <p className="text-sm text-muted-foreground">설명력</p>
              </div>
              <div className="bg-background border p-4 rounded-lg">
                <h4 className="font-medium text-foreground">MAE</h4>
                <p className="text-2xl font-bold text-primary">
                  {formatNumber(modelPerformance.performance.train_metrics?.mae || 0, 3)}
                </p>
                <p className="text-sm text-muted-foreground">평균 절대 오차</p>
              </div>
              <div className="bg-background border p-4 rounded-lg">
                <h4 className="font-medium text-foreground">RMSE</h4>
                <p className="text-2xl font-bold text-primary">
                  {formatNumber(Math.sqrt(modelPerformance.performance.train_metrics?.mse || 0), 3)}
                </p>
                <p className="text-sm text-muted-foreground">평균 제곱근 오차</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 분석 도구 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SHAP 분석 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="mr-2 h-5 w-5" />
              SHAP 분석
              {shapAnalysis?.available && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              SHAP 값을 통한 feature 중요도 분석
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
                  <div className="max-h-48 overflow-y-auto">
                    {shapAnalysis.feature_importance.slice(0, 10).map((item, idx) => (
                      <div key={idx} className="flex justify-between items-center py-1">
                        <span className="text-sm">{item.feature}</span>
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

        {/* LIME 분석 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Eye className="mr-2 h-5 w-5" />
              LIME 분석
              {limeAnalysis?.available && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              개별 예측에 대한 설명
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleLimeAnalysis}
              disabled={loading === 'lime'}
            >
              {loading === 'lime' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  분석 중...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  LIME 분석 시작
                </>
              )}
            </Button>

            {limeAnalysis?.error ? (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{limeAnalysis.error}</AlertDescription>
              </Alert>
            ) : limeAnalysis?.available ? (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  예측값: {formatNumber(limeAnalysis.prediction, 3)}
                </p>
                {limeAnalysis.explanation && (
                  <div className="max-h-48 overflow-y-auto">
                    {limeAnalysis.explanation.slice(0, 8).map((item: any, idx: number) => (
                      <div key={idx} className="flex justify-between items-center py-1">
                        <span className="text-sm">{item.feature}</span>
                        <span className={`text-sm font-medium ${item.value > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {item.value > 0 ? '+' : ''}{formatNumber(item.value, 4)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                샘플을 선택하고 분석을 시작하세요
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Feature Importance 차트 */}
      {featureImportance.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="mr-2 h-5 w-5" />
              Feature 중요도
            </CardTitle>
            <CardDescription>
              전체 모델에서의 각 변수의 상대적 중요도
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {featureImportance.slice(0, 15).map((item, idx) => {
                const maxImportance = Math.max(...featureImportance.map(f => f.importance));
                const width = (item.importance / maxImportance) * 100;
                
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{item.feature}</span>
                      <span className="text-sm text-muted-foreground">
                        {formatNumber(item.importance, 4)}
                        {item.std && ` ± ${formatNumber(item.std, 4)}`}
                      </span>
                    </div>
                    <div className="w-full bg-border rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-300"
                        style={{ width: `${width}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};