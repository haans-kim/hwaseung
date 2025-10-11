import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Users,
  Target,
  AlertTriangle,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { API_BASE_URL } from '../lib/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ORGANIZATION = 'R&A';

interface PredictionData {
  year: number;
  predicted_headcount: number;
  previous_headcount: number;
  change: number;
  change_percent: number;
  model_r2: number;
}

interface FeatureImportance {
  feature: string;  // 백엔드 API는 'feature' 키 사용
  importance: number;
  label: string;
  std?: number;
}

interface TrendData {
  years: number[];
  actual: (number | null)[];
  predicted: (number | null)[];
}

interface SimulationResult {
  predicted_headcount: number;
  baseline_headcount: number;
  change: number;
}

const DashboardRNA: React.FC = () => {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [importance, setImportance] = useState<FeatureImportance[]>([]);
  const [trendData, setTrendData] = useState<TrendData | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [variables, setVariables] = useState<{[key: string]: number}>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Feature 한글 라벨
  const featureLabels: {[key: string]: string} = {
    'ev_growth_gl': '글로벌 EV시장 성장률',
    'v_growth_gl': '글로벌 자동차 시장성장률',
    'v_export_kr': '국내 자동차 수출액 증가율',
    'vp_export_kr': '국내 자동차부품 수출액 증가율',
    'gdp_growth_kr': 'GDP성장률',
    'cpi_kr': '소비자물가상승률',
    'exchange_rate_change_krw': '환율변화율',
    'scm_index_gl': '글로벌물류비지수',
    'oil_gl': '국제유가',
    'labor_cost': '인건비 증감률',
    'revenue': '매출액 증감률',
    'profit': '영업이익 증감률',
    'operating_rate': '가동률 증감률',
    'operating_date': '가동일수 증감률'
  };

  // 초기 데이터 로드
  useEffect(() => {
    loadAllData();
  }, []);

  const loadAllData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 병렬로 모든 데이터 로드
      const [predRes, impRes, trendRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/company-wide/dashboard/prediction?organization=${ORGANIZATION}`),
        fetch(`${API_BASE_URL}/api/company-wide/dashboard/importance?organization=${ORGANIZATION}`),
        fetch(`${API_BASE_URL}/api/company-wide/dashboard/trend?organization=${ORGANIZATION}`)
      ]);

      if (predRes.ok) {
        const predData = await predRes.json();
        setPrediction(predData);
      }

      if (impRes.ok) {
        const impData = await impRes.json();
        const featuresWithLabels = impData.features.map((f: FeatureImportance) => ({
          ...f,
          label: featureLabels[f.name] || f.name
        }));
        setImportance(featuresWithLabels);

        // 초기 변수값 설정 (시뮬레이션용)
        const initialVars: {[key: string]: number} = {};
        featuresWithLabels.forEach((f: FeatureImportance) => {
          initialVars[f.name] = 0; // 기본값 0
        });
        setVariables(initialVars);
      }

      if (trendRes.ok) {
        const trend = await trendRes.json();
        setTrendData(trend);
      }

    } catch (err: any) {
      setError(err.message || '데이터 로드 실패');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // 시나리오 시뮬레이션
  const handleSimulate = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/company-wide/dashboard/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          organization: ORGANIZATION,
          variables
        })
      });

      if (!response.ok) {
        throw new Error('시뮬레이션 실패');
      }

      const data = await response.json();
      setSimulationResult(data);
    } catch (err: any) {
      setError(err.message);
      console.error(err);
    }
  };

  // 변수 리셋
  const handleReset = () => {
    const resetVars: {[key: string]: number} = {};
    Object.keys(variables).forEach(key => {
      resetVars[key] = 0;
    });
    setVariables(resetVars);
    setSimulationResult(null);
  };

  // 트렌드 차트 데이터
  const trendChartData = trendData ? {
    labels: trendData.years.map(y => `${y}년`),
    datasets: [
      {
        label: '실제 정원',
        data: trendData.actual,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      },
      {
        label: '2026년 예측',
        data: trendData.predicted,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        tension: 0.4
      }
    ]
  } : null;

  // Feature Importance 차트 데이터
  const importanceChartData = importance.length > 0 ? {
    labels: importance.slice(0, 10).map(f => f.label || f.name),
    datasets: [{
      label: 'Importance',
      data: importance.slice(0, 10).map(f => f.importance),
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
      borderColor: 'rgb(59, 130, 246)',
      borderWidth: 1
    }]
  } : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">R&A 적정인력 산정</h1>
          <p className="text-muted-foreground">2026년 적정인력 예측 및 시나리오 분석</p>
        </div>
        <Button onClick={loadAllData} variant="outline">
          <RefreshCw className="mr-2 h-4 w-4" />
          새로고침
        </Button>
      </div>

      {/* 에러 */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 주요 지표 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* 2026년 예측 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">2026년 적정인력</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {prediction ? `${Math.round(prediction.predicted_headcount)}명` : '-'}
            </div>
            <p className="text-xs text-muted-foreground">
              예측 정원
            </p>
          </CardContent>
        </Card>

        {/* 증감 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">전년 대비</CardTitle>
            {prediction && prediction.change < 0 ? (
              <TrendingDown className="h-4 w-4 text-red-500" />
            ) : (
              <TrendingUp className="h-4 w-4 text-green-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${prediction && prediction.change < 0 ? 'text-red-500' : 'text-green-500'}`}>
              {prediction ? `${prediction.change > 0 ? '+' : ''}${Math.round(prediction.change)}명` : '-'}
            </div>
            <p className="text-xs text-muted-foreground">
              {prediction ? `${prediction.change_percent.toFixed(1)}%` : '-'}
            </p>
          </CardContent>
        </Card>

        {/* 모델 정확도 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">모델 정확도</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {prediction && prediction.model_r2 != null ? `R² ${prediction.model_r2.toFixed(3)}` : '-'}
            </div>
            <p className="text-xs text-muted-foreground">
              결정계수
            </p>
          </CardContent>
        </Card>

        {/* 시뮬레이션 결과 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">시나리오 예측</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {simulationResult ? `${Math.round(simulationResult.predicted_headcount)}명` : '-'}
            </div>
            <p className="text-xs text-muted-foreground">
              {simulationResult ? `${simulationResult.change > 0 ? '+' : ''}${Math.round(simulationResult.change)}명` : '변수 조정 필요'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 트렌드 분석 */}
      {trendChartData && (
        <Card>
          <CardHeader>
            <CardTitle>인력 추이 및 2026년 예측</CardTitle>
            <CardDescription>과거 실적과 미래 예측</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <Line
                data={trendChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'top' as const },
                    tooltip: {
                      callbacks: {
                        label: (context) => {
                          return `${context.dataset.label}: ${context.parsed.y ? Math.round(context.parsed.y) : '-'}명`;
                        }
                      }
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: false,
                      ticks: {
                        callback: (value) => `${value}명`
                      }
                    }
                  }
                }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* 변수 조정 */}
      <Card>
        <CardHeader>
          <CardTitle>변수 조정 시뮬레이션</CardTitle>
          <CardDescription>주요 변수를 조정하여 시나리오별 예측 확인</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {importance.slice(0, 8).map((feature) => (
              <div key={feature.name} className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">{feature.label}</label>
                  <span className="text-sm text-muted-foreground">{variables[feature.name] || 0}%</span>
                </div>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  step="1"
                  value={variables[feature.name] || 0}
                  onChange={(e) => setVariables({...variables, [feature.name]: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>
            ))}
          </div>

          <div className="flex gap-2">
            <Button onClick={handleSimulate} className="flex-1">
              <Target className="mr-2 h-4 w-4" />
              시뮬레이션 실행
            </Button>
            <Button onClick={handleReset} variant="outline">
              <RefreshCw className="mr-2 h-4 w-4" />
              초기화
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      {importanceChartData && (
        <Card>
          <CardHeader>
            <CardTitle>영향 요인 분석</CardTitle>
            <CardDescription>적정인력에 영향을 미치는 주요 변수 (Permutation Importance)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <Bar
                data={importanceChartData}
                options={{
                  indexAxis: 'y' as const,
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    x: {
                      beginAtZero: true,
                      ticks: {
                        callback: (value) => typeof value === 'number' ? value.toFixed(3) : value
                      }
                    }
                  }
                }}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* 모델 학습 안내 */}
      {!prediction && !loading && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>모델이 학습되지 않았습니다</AlertTitle>
          <AlertDescription>
            <div className="mt-2">
              <p>R&A 적정인력 예측을 위해 먼저 모델을 학습해주세요.</p>
              <Button
                className="mt-2"
                onClick={() => window.location.href = '/company-wide-modeling'}
              >
                모델링 페이지로 이동
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default DashboardRNA;
