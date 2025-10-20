import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { apiClient } from '../lib/api';

interface OrganizationData {
  회사: string;
  본부: string;
  담당_사업단_센터: string | null;
  실: string | null;
  팀: string | null;
}

interface RegressionModel {
  id: number;
  org_name: string;
  model_type: string;
}

interface RegressionParameter {
  id: number;
  model_id: number;
  parameter_name: string;
  coefficient: number;
}

interface TeamMetric {
  team_name: string;
  metric_name: string;
  avg_value: number;
}

const OrganizationSimulation: React.FC = () => {
  const [organizationData, setOrganizationData] = useState<OrganizationData[]>([]);
  const [selectedCompany, setSelectedCompany] = useState<string | null>('화승 R&A');
  const [selectedDepartment, setSelectedDepartment] = useState<string | null>(null);
  const [selectedDivision, setSelectedDivision] = useState<string | null>(null);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  const [departments, setDepartments] = useState<string[]>([]);
  const [divisions, setDivisions] = useState<string[]>([]);
  const [sections, setSections] = useState<string[]>([]);
  const [teams, setTeams] = useState<string[]>([]);

  const [availableRegressionTeams, setAvailableRegressionTeams] = useState<string[]>([]);

  // 4개 모델 관리 (총, 책임, 선임, 사원)
  const [regressionModels, setRegressionModels] = useState<{ [key: string]: RegressionModel }>({});
  const [regressionParameters, setRegressionParameters] = useState<{ [key: string]: RegressionParameter[] }>({});
  const [teamMetrics, setTeamMetrics] = useState<{ [key: string]: number }>({});
  const [adjustedMetrics, setAdjustedMetrics] = useState<{ [key: string]: number }>({});
  const [featureDefinitions, setFeatureDefinitions] = useState<{ [key: string]: string }>({});

  // 4개 예측 결과 (총, 책임, 선임, 사원)
  const [currentHeadcount, setCurrentHeadcount] = useState<{ [key: string]: number }>({});
  const [predictedHeadcount, setPredictedHeadcount] = useState<{ [key: string]: number }>({});
  const [currentFTE, setCurrentFTE] = useState<{ [key: string]: number }>({});

  // 기간 정보 (동적 년월)
  const [currentPeriod, setCurrentPeriod] = useState<string | null>(null);
  const [predictionPeriod, setPredictionPeriod] = useState<string | null>(null);

  // API로 데이터 로드
  useEffect(() => {
    const loadData = async () => {
      try {
        // 1. 조직 데이터 로드 (API)
        const orgResponse = await apiClient.getOrganizationChartData();
        console.log('📊 Organization data from API:', orgResponse);

        if (orgResponse && orgResponse.data) {
          const orgData = orgResponse.data.map((row: any) => ({
            회사: row.회사,
            본부: row.본부,
            담당_사업단_센터: row.담당_사업단_센터,
            실: row.실,
            팀: row.팀,
          }));
          setOrganizationData(orgData);

          // 초기 본부 설정
          if (orgData.length > 0) {
            const uniqueDepartments = Array.from(new Set(
              orgData
                .filter((org: any) => org.회사 === '화승 R&A')
                .map((org: any) => org.본부)
            )) as string[];
            setDepartments(uniqueDepartments);
          }
        }

        // 2. 분석가능팀 목록 로드 (API)
        const teamsResponse = await apiClient.getAnalysisReadyTeams();
        console.log('🌐 Analysis-ready teams from API:', teamsResponse);

        if (teamsResponse && teamsResponse.teams && teamsResponse.teams.length > 0) {
          const teams = teamsResponse.teams.map((team: { team: string }) => team.team);
          console.log('✅ Analysis-ready teams:', teams);
          setAvailableRegressionTeams(teams);
        }

      } catch (error) {
        console.error('❌ Error loading data from API:', error);
      }
    };

    loadData();
  }, []);

  // 회사 선택 시 본부 목록 업데이트
  useEffect(() => {
    if (selectedCompany && organizationData.length > 0) {
      const uniqueDepartments = Array.from(new Set(
        organizationData
          .filter(org => org.회사 === selectedCompany)
          .map(org => org.본부)
      ));
      setDepartments(uniqueDepartments);
      setSelectedDepartment(null);  // 회사 변경시 본부 선택 초기화
    }
  }, [selectedCompany, organizationData]);

  // 부서별 조직 구조 업데이트
  useEffect(() => {
    if (selectedCompany && selectedDepartment) {
      const uniqueDivisions = Array.from(
        new Set(
          organizationData
            .filter(
              org =>
                org.회사 === selectedCompany &&
                org.본부 === selectedDepartment &&
                org.담당_사업단_센터
            )
            .map(org => org.담당_사업단_센터!)
        )
      );
      setDivisions(uniqueDivisions);
    } else {
      setDivisions([]);
    }
    setSelectedDivision(null);
  }, [selectedCompany, selectedDepartment, organizationData]);

  useEffect(() => {
    if (selectedCompany && selectedDepartment && selectedDivision) {
      const uniqueSections = Array.from(
        new Set(
          organizationData
            .filter(
              org =>
                org.회사 === selectedCompany &&
                org.본부 === selectedDepartment &&
                org.담당_사업단_센터 === selectedDivision &&
                org.실
            )
            .map(org => org.실!)
        )
      );
      setSections(uniqueSections);
    } else {
      setSections([]);
    }
    setSelectedSection(null);
  }, [selectedCompany, selectedDepartment, selectedDivision, organizationData]);

  useEffect(() => {
    if (selectedCompany && selectedDepartment && selectedDivision && selectedSection) {
      const uniqueTeams = Array.from(
        new Set(
          organizationData
            .filter(
              org =>
                org.회사 === selectedCompany &&
                org.본부 === selectedDepartment &&
                org.담당_사업단_센터 === selectedDivision &&
                org.실 === selectedSection &&
                org.팀
            )
            .map(org => org.팀!)
        )
      );
      setTeams(uniqueTeams);
    } else {
      setTeams([]);
    }
    setSelectedTeam(null);
  }, [selectedCompany, selectedDepartment, selectedDivision, selectedSection, organizationData]);

  // 팀 선택 시 4개 모델 로드
  useEffect(() => {
    if (selectedTeam && availableRegressionTeams.includes(selectedTeam)) {
      loadTeam4Models(selectedTeam);
    }
  }, [selectedTeam, availableRegressionTeams]);

  const loadTeam4Models = async (teamName: string) => {
    try {
      // API로 팀 시뮬레이션 데이터 조회
      const response = await apiClient.getTeamSimulationData(teamName);
      console.log('📊 Team simulation data from API:', response);

      if (!response || !response.data) {
        console.error('No simulation data available for team:', teamName);
        return;
      }

      const data = response.data;

      // 1. Regression Models & Parameters 설정
      const models: { [key: string]: RegressionModel } = {};
      const allParameters: { [key: string]: RegressionParameter[] } = {};

      Object.entries(data.regression_models).forEach(([modelType, modelData]: [string, any]) => {
        models[modelType] = {
          id: modelData.id,
          org_name: modelData.org_name,
          model_type: modelData.model_type,
        };

        // Parameters를 배열로 변환
        const params: RegressionParameter[] = [];
        Object.entries(modelData.parameters).forEach(([paramName, coefficient]: [string, any]) => {
          params.push({
            id: 0,
            model_id: modelData.id,
            parameter_name: paramName,
            coefficient: coefficient,
          });
        });
        allParameters[modelType] = params;
      });

      setRegressionModels(models);
      setRegressionParameters(allParameters);

      // 2. Team Metrics 설정
      // team_metrics가 비어있으면 feature_definitions를 기반으로 기본값 0으로 초기화
      const metrics = Object.keys(data.team_metrics).length > 0
        ? data.team_metrics
        : Object.keys(data.feature_definitions || {}).reduce((acc, key) => {
            acc[key] = 0;
            return acc;
          }, {} as { [key: string]: number });

      setTeamMetrics(metrics);
      setAdjustedMetrics(metrics);

      // 2-1. Feature Definitions 설정 (F1, F2 -> 실제 이름)
      setFeatureDefinitions(data.feature_definitions || {});

      // 3. Current Headcount 설정
      setCurrentHeadcount(data.current_headcount);

      // 4. Current FTE 설정
      setCurrentFTE(data.current_fte);

      // 5. 기간 정보 설정 (동적 년월)
      setCurrentPeriod(data.current_period);
      setPredictionPeriod(data.prediction_period);

      console.log('✅ Team data loaded successfully from API');
      console.log('📅 Current period:', data.current_period, '| Prediction period:', data.prediction_period);

    } catch (error) {
      console.error('❌ Error loading team data from API:', error);
    }
  };

  // 4개 예측 인원 계산
  useEffect(() => {
    if (Object.keys(regressionParameters).length > 0 && Object.keys(adjustedMetrics).length > 0) {
      calculate4Predictions();
    }
  }, [regressionParameters, adjustedMetrics]);

  const calculate4Predictions = () => {
    const predictions: { [key: string]: number } = {};
    const modelTypes = ['총', '책임', '선임', '사원'];

    // // console.log('Calculating predictions with adjustedMetrics:', adjustedMetrics);
    // // console.log('Regression parameters available for:', Object.keys(regressionParameters));

    modelTypes.forEach(modelType => {
      if (regressionParameters[modelType]) {
        let prediction = 0;
        const intercept = regressionParameters[modelType].find(p => p.parameter_name === 'intercept');
        if (intercept) {
          prediction = intercept.coefficient;
          // // console.log(`${modelType} - intercept: ${intercept.coefficient}`);
        }

        regressionParameters[modelType].forEach(param => {
          if (param.parameter_name !== 'intercept' && adjustedMetrics[param.parameter_name]) {
            const contribution = param.coefficient * adjustedMetrics[param.parameter_name];
            prediction += contribution;
            // console.log(`${modelType} - ${param.parameter_name}: ${param.coefficient} * ${adjustedMetrics[param.parameter_name]} = ${contribution}`);
          }
        });

        prediction = prediction * 1.0; // 과적합 조정 제거
        predictions[modelType] = Math.max(0, Math.round(prediction));
        // console.log(`${modelType} final prediction: ${predictions[modelType]}`);
      } else {
        // console.log(`No regression parameters found for ${modelType}`);
      }
    });

    // console.log('Final predictions:', predictions);
    setPredictedHeadcount(predictions);
  };

  const handleMetricChange = (metricName: string, value: number) => {
    const baseValue = teamMetrics[metricName] || 0;
    const adjustedValue = baseValue * (1 + value / 100);
    // console.log(`Adjusting ${metricName}: base=${baseValue}, adjustment=${value}%, new=${adjustedValue}`);
    setAdjustedMetrics(prev => ({
      ...prev,
      [metricName]: adjustedValue
    }));
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-blue-600';
    if (change < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">조직, 직급별 적정인력 산정</h1>
          <p className="text-muted-foreground">조직, 직급별 적정인력 예측 및 Simulation</p>
        </div>
      </div>

      {/* Miller Column Navigation */}
      <div className="grid grid-cols-6 gap-4">
        {/* 회사 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">회사</h3>
          <div className="space-y-2">
            <button
              onClick={() => setSelectedCompany('화승 R&A')}
              className={`w-full px-3 py-2 text-left rounded text-sm ${
                selectedCompany === '화승 R&A'
                  ? 'bg-blue-100 text-blue-800 border border-blue-300'
                  : 'hover:bg-gray-50 border border-gray-200'
              }`}
            >
              화승 R&A
            </button>
            <button
              onClick={() => setSelectedCompany('화승 Corp.')}
              className={`w-full px-3 py-2 text-left rounded text-sm ${
                selectedCompany === '화승 Corp.'
                  ? 'bg-blue-100 text-blue-800 border border-blue-300'
                  : 'hover:bg-gray-50 border border-gray-200'
              }`}
            >
              화승 Corp.
            </button>
          </div>
        </Card>

        {/* 본부 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">본부</h3>
          <div className="space-y-2">
            {departments.map((dept) => (
              <button
                key={dept}
                onClick={() => setSelectedDepartment(dept)}
                className={`w-full px-3 py-2 text-left rounded text-sm ${
                  selectedDepartment === dept
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'hover:bg-gray-50 border border-gray-200'
                }`}
              >
                {dept}
              </button>
            ))}
          </div>
        </Card>

        {/* 담당 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">담당</h3>
          <div className="space-y-2">
            {divisions.map((div) => (
              <button
                key={div}
                onClick={() => setSelectedDivision(div)}
                className={`w-full px-3 py-2 text-left rounded text-sm ${
                  selectedDivision === div
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'hover:bg-gray-50 border border-gray-200'
                }`}
              >
                {div}
              </button>
            ))}
          </div>
        </Card>

        {/* 실 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">실</h3>
          <div className="space-y-2">
            {sections.map((section) => (
              <button
                key={section}
                onClick={() => setSelectedSection(section)}
                className={`w-full px-3 py-2 text-left rounded text-sm ${
                  selectedSection === section
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'hover:bg-gray-50 border border-gray-200'
                }`}
              >
                {section}
              </button>
            ))}
          </div>
        </Card>

        {/* 팀 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">팀</h3>
          <div className="space-y-2">
            {teams.map((team) => {
              const hasData = availableRegressionTeams.includes(team);
              return (
                <button
                  key={team}
                  onClick={() => setSelectedTeam(team)}
                  className={`w-full px-3 py-2 text-left rounded text-sm ${
                    selectedTeam === team
                      ? 'bg-blue-100 text-blue-800 border border-blue-300'
                      : hasData
                        ? 'bg-gray-200 hover:bg-gray-300 border border-gray-300'
                        : 'bg-white hover:bg-gray-50 border border-gray-200'
                  }`}
                >
                  {team}
                </button>
              );
            })}
          </div>
        </Card>

        {/* 분석가능팀 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">분석가능팀</h3>
          <div className="space-y-2 max-h-96 overflow-y-scroll scrollbar-visible">
            {availableRegressionTeams.map((team) => (
              <button
                key={team}
                onClick={() => setSelectedTeam(team)}
                className={`w-full px-3 py-2 text-left rounded text-sm ${
                  selectedTeam === team
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'hover:bg-gray-50 border border-gray-200'
                }`}
              >
                {team}
              </button>
            ))}
          </div>
        </Card>
      </div>

      {/* 전체 인력 변동 요약 카드 */}
      {selectedTeam && availableRegressionTeams.includes(selectedTeam) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* 현재 정원 */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600">{currentPeriod || '현재'} 정원</h3>
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-blue-600 text-sm font-semibold">👥</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {(currentHeadcount['총'] || 0)}명
            </div>
            <div className="text-xs text-green-600">현재 인원</div>
          </div>

          {/* 예상 정원 */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600">{predictionPeriod || '다음'} 예상 정원</h3>
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                <span className="text-purple-600 text-sm font-semibold">🎯</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {(predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)}명
            </div>
            <div className={`text-xs ${
              ((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) > 0
                ? 'text-green-600'
                : ((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) < 0
                  ? 'text-red-600'
                  : 'text-gray-600'
            }`}>
              {((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) > 0 ? '+' : ''}
              {((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0)}명 변화
            </div>
          </div>

          {/* 변화율 */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600">변화율</h3>
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                <span className="text-orange-600 text-sm font-semibold">📈</span>
              </div>
            </div>
            <div className={`text-2xl font-bold mb-1 ${
              ((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) > 0
                ? 'text-green-600'
                : ((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) < 0
                  ? 'text-red-600'
                  : 'text-gray-600'
            }`}>
              {currentHeadcount['총'] > 0
                ? `${((((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0)) / (currentHeadcount['총'] || 1) * 100).toFixed(1)}%`
                : '0%'
              }
            </div>
            <div className="text-xs text-gray-600">예상 증감률</div>
          </div>
        </div>
      )}

      {/* 시뮬레이션 패널 */}
      {selectedTeam && availableRegressionTeams.includes(selectedTeam) && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 슬라이더 패널 */}
          <div className="lg:col-span-2">
            <Card className="p-6 bg-white">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">업무 지표 조정</h3>
              <div className="space-y-4">
                {Object.entries(teamMetrics).map(([metricName, baseValue]) => {
                  const adjustment = ((adjustedMetrics[metricName] || baseValue) - baseValue) / baseValue * 100;
                  // F1, F2 등을 실제 feature 이름으로 변환
                  const displayName = featureDefinitions[metricName] || metricName;
                  return (
                    <div key={metricName} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">{displayName}</span>
                        <span className="text-sm text-gray-500">
                          기준: {baseValue.toFixed(1)} | 현재: {adjustedMetrics[metricName]?.toFixed(1) || baseValue.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className="text-xs text-gray-500 w-8">-50%</span>
                        <input
                          type="range"
                          min="-50"
                          max="50"
                          step="1"
                          value={adjustment}
                          onChange={(e) => handleMetricChange(metricName, Number(e.target.value))}
                          className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <span className="text-xs text-gray-500 w-8">+50%</span>
                        <span className={`text-xs font-medium w-12 text-right ${getChangeColor(adjustment)}`}>
                          {adjustment > 0 ? '+' : ''}{adjustment.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>

          {/* 예측 결과 테이블 */}
          <div>
            <Card className="p-6 bg-white">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">인력 예측 결과</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-3 font-medium text-gray-700">구분</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">현재<br/>인원</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">현재<br/>FTE</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">예측<br/>인원</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">변화</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-blue-700">전체</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['총'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-blue-600">{(currentFTE['총'] || 0).toFixed(1)}</td>
                      <td className="text-center py-2 px-3 text-blue-600 font-semibold">
                        {(predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)}명
                      </td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor(
                        ((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0)
                      )}`}>
                        {((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0) > 0 ? '+' : ''}
                        {((predictedHeadcount['책임'] || 0) + (predictedHeadcount['선임'] || 0) + (predictedHeadcount['사원'] || 0)) - (currentHeadcount['총'] || 0)}명
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-green-700">책임</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['책임'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-green-600">{(currentFTE['책임'] || 0).toFixed(1)}</td>
                      <td className="text-center py-2 px-3 text-green-600 font-semibold">{predictedHeadcount['책임'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0))}`}>
                        {(predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0)}명
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-orange-700">선임</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['선임'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-orange-600">{(currentFTE['선임'] || 0).toFixed(1)}</td>
                      <td className="text-center py-2 px-3 text-orange-600 font-semibold">{predictedHeadcount['선임'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0))}`}>
                        {(predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0)}명
                      </td>
                    </tr>
                    <tr className="hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-purple-700">사원</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['사원'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-purple-600">{(currentFTE['사원'] || 0).toFixed(1)}</td>
                      <td className="text-center py-2 px-3 text-purple-600 font-semibold">{predictedHeadcount['사원'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['사원'] || 0) - (currentHeadcount['사원'] || 0))}`}>
                        {(predictedHeadcount['사원'] || 0) - (currentHeadcount['사원'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['사원'] || 0) - (currentHeadcount['사원'] || 0)}명
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Card>

            {/* 디버깅 정보 패널 */}
            <Card className="p-6 bg-gray-50 border-2 border-gray-300 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-gray-800">
                🔍 회귀 분석 디버깅 정보
              </h3>

              {['총', '책임', '선임', '사원'].map((position) => {
                const paramsArray = regressionParameters[position];
                if (!paramsArray || paramsArray.length === 0) return null;

                // intercept 찾기
                const interceptParam = paramsArray.find(p => p.parameter_name === 'intercept');
                const intercept = interceptParam?.coefficient || 0;

                // intercept를 제외한 다른 파라미터들
                const otherParams = paramsArray.filter(p => p.parameter_name !== 'intercept');

                // 계산 과정
                let calculation = intercept;
                const terms: string[] = [];

                return (
                  <div key={position} className="mb-6 p-4 bg-white rounded-lg border border-gray-200">
                    <h4 className={`font-semibold mb-3 ${
                      position === '총' ? 'text-blue-700' :
                      position === '책임' ? 'text-green-700' :
                      position === '선임' ? 'text-orange-700' :
                      'text-purple-700'
                    }`}>
                      {position} 모델
                    </h4>

                    <div className="grid grid-cols-2 gap-4 mb-3">
                      <div>
                        <span className="font-medium">Y절편 (Intercept):</span>
                        <span className="ml-2 font-mono">{intercept.toFixed(4)}</span>
                      </div>
                    </div>

                    <div className="mb-3">
                      <span className="font-medium">회귀 계수:</span>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                        {otherParams.map((param) => {
                          const coef = param.coefficient;
                          const value = adjustedMetrics[param.parameter_name] || 0;
                          const contribution = coef * value;

                          if (coef !== 0) {
                            terms.push(`${coef.toFixed(5)} × ${value.toFixed(2)}`);
                            calculation += contribution;
                          }

                          return (
                            <div key={param.parameter_name} className="font-mono p-2 bg-gray-50 rounded">
                              <div className="text-xs text-gray-600">{param.parameter_name}:</div>
                              <div className={coef !== 0 ? 'text-blue-600' : 'text-gray-400'}>
                                계수: {coef.toFixed(5)}
                              </div>
                              <div className="text-xs">
                                현재값: {value.toFixed(2)}
                              </div>
                              {coef !== 0 && (
                                <div className="text-xs text-green-600">
                                  기여도: {contribution.toFixed(4)}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="p-3 bg-blue-50 rounded-lg">
                      <div className="font-medium mb-2">계산식:</div>
                      <div className="font-mono text-sm break-all">
                        Y = {intercept.toFixed(4)}
                        {terms.length > 0 && ' + '}
                        {terms.join(' + ')}
                      </div>
                      <div className="mt-2 font-semibold text-blue-700">
                        = {calculation.toFixed(4)} → 반올림: {Math.round(calculation)}명
                      </div>
                      <div className="mt-1 text-sm text-gray-600">
                        실제 예측값: {predictedHeadcount[position] || 0}명
                      </div>
                    </div>
                  </div>
                );
              })}
            </Card>
          </div>
        </div>
      )}

      {!selectedTeam && (
        <Card className="p-8 bg-white text-center">
          <p className="text-gray-500">시뮬레이션을 시작하려면 팀을 선택해주세요</p>
        </Card>
      )}
    </div>
  );
};

export default OrganizationSimulation;