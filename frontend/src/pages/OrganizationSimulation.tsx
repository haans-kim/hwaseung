import React, { useState, useEffect } from 'react';
import initSqlJs from 'sql.js';
import { Card } from '../components/ui/card';

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
  const [selectedCompany, setSelectedCompany] = useState<string | null>('화승R&A');
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

  // 4개 예측 결과 (총, 책임, 선임, 사원)
  const [currentHeadcount, setCurrentHeadcount] = useState<{ [key: string]: number }>({});
  const [predictedHeadcount, setPredictedHeadcount] = useState<{ [key: string]: number }>({});

  // SQLite 데이터 로드
  useEffect(() => {
    const loadData = async () => {
      try {
        const SQL = await initSqlJs({
          locateFile: (file: string) => `https://sql.js.org/dist/${file}`
        });

        const response = await fetch('/hwaseung_RnD.db');
        const buffer = await response.arrayBuffer();
        const db = new SQL.Database(new Uint8Array(buffer));

        // 조직 데이터 로드
        const orgResult = db.exec('SELECT * FROM organization');
        if (orgResult.length > 0) {
          const orgData = orgResult[0].values.map(row => ({
            회사: row[1] as string,
            본부: row[2] as string,
            담당_사업단_센터: row[3] as string | null,
            실: row[4] as string | null,
            팀: row[5] as string | null,
          }));
          setOrganizationData(orgData);

          // 초기 본부 설정
          if (orgData.length > 0) {
            const uniqueDepartments = Array.from(new Set(
              orgData
                .filter(org => org.회사 === '화승R&A')
                .map(org => org.본부)
            ));
            setDepartments(uniqueDepartments);
          }
        }

        // 회귀 모델이 있는 팀 목록 로드
        const regressionTeamsResult = db.exec(`
          SELECT DISTINCT org_name
          FROM regression_models
          WHERE model_type IN ('총', '책임', '선임', '사원')
          ORDER BY org_name
        `);

        if (regressionTeamsResult.length > 0) {
          const teams = regressionTeamsResult[0].values.map(row => row[0] as string);
          setAvailableRegressionTeams(teams);
        }

        db.close();
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };

    loadData();
  }, []);

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
      const SQL = await initSqlJs({
        locateFile: (file: string) => `https://sql.js.org/dist/${file}`
      });

      const response = await fetch('/hwaseung_RnD.db');
      const buffer = await response.arrayBuffer();
      const db = new SQL.Database(new Uint8Array(buffer));

      const models: { [key: string]: RegressionModel } = {};
      const allParameters: { [key: string]: RegressionParameter[] } = {};
      const modelTypes = ['총', '책임', '선임', '사원'];

      // 4개 모델 로드
      for (const modelType of modelTypes) {
        const modelResult = db.exec(`
          SELECT * FROM regression_models
          WHERE org_name = '${teamName}' AND model_type = '${modelType}'
          LIMIT 1
        `);

        if (modelResult.length > 0 && modelResult[0].values.length > 0) {
          const model: RegressionModel = {
            id: modelResult[0].values[0][0] as number,
            org_name: modelResult[0].values[0][1] as string,
            model_type: modelResult[0].values[0][2] as string,
          };
          models[modelType] = model;

          const paramResult = db.exec(`
            SELECT * FROM regression_parameters
            WHERE model_id = ${model.id}
          `);

          if (paramResult.length > 0) {
            const params = paramResult[0].values.map(row => ({
              id: row[0] as number,
              model_id: row[1] as number,
              parameter_name: row[2] as string,
              coefficient: row[3] as number,
            }));
            allParameters[modelType] = params;
          }
        }
      }

      setRegressionModels(models);
      setRegressionParameters(allParameters);

      // 팀 메트릭 평균값 로드
      const metricResult = db.exec(`
        SELECT metric_name, AVG(metric_value) as avg_value
        FROM team_metrics
        WHERE team_name = '${teamName}'
        GROUP BY metric_name
      `);

      if (metricResult.length > 0) {
        const metrics: { [key: string]: number } = {};
        metricResult[0].values.forEach(row => {
          metrics[row[0] as string] = row[1] as number;
        });
        setTeamMetrics(metrics);
        setAdjustedMetrics(metrics);
      }

      // 현재 인원 로드 (team_headcount 테이블에서 25년 8월 최신 데이터)
      const currentHeadcountData: { [key: string]: number } = {};

      const totalResult = db.exec(`
        SELECT headcount FROM team_headcount
        WHERE team_name = '${teamName}' AND year = 25 AND month = 8 AND position = '총합'
        LIMIT 1
      `);
      if (totalResult.length > 0 && totalResult[0].values.length > 0) {
        currentHeadcountData['총'] = totalResult[0].values[0][0] as number;
      }

      const managerResult = db.exec(`
        SELECT headcount FROM team_headcount
        WHERE team_name = '${teamName}' AND year = 25 AND month = 8 AND position = '책임'
        LIMIT 1
      `);
      if (managerResult.length > 0 && managerResult[0].values.length > 0) {
        currentHeadcountData['책임'] = managerResult[0].values[0][0] as number;
      }

      const seniorResult = db.exec(`
        SELECT headcount FROM team_headcount
        WHERE team_name = '${teamName}' AND year = 25 AND month = 8 AND position = '선임'
        LIMIT 1
      `);
      if (seniorResult.length > 0 && seniorResult[0].values.length > 0) {
        currentHeadcountData['선임'] = seniorResult[0].values[0][0] as number;
      }

      const juniorResult = db.exec(`
        SELECT headcount FROM team_headcount
        WHERE team_name = '${teamName}' AND year = 25 AND month = 8 AND position = '사원'
        LIMIT 1
      `);
      if (juniorResult.length > 0 && juniorResult[0].values.length > 0) {
        currentHeadcountData['사원'] = juniorResult[0].values[0][0] as number;
      }

      setCurrentHeadcount(currentHeadcountData);

      db.close();
    } catch (error) {
      console.error('Error loading team 4 models:', error);
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

    console.log('Calculating predictions with adjustedMetrics:', adjustedMetrics);
    console.log('Regression parameters available for:', Object.keys(regressionParameters));

    modelTypes.forEach(modelType => {
      if (regressionParameters[modelType]) {
        let prediction = 0;
        const intercept = regressionParameters[modelType].find(p => p.parameter_name === 'intercept');
        if (intercept) {
          prediction = intercept.coefficient;
          console.log(`${modelType} - intercept: ${intercept.coefficient}`);
        }

        regressionParameters[modelType].forEach(param => {
          if (param.parameter_name !== 'intercept' && adjustedMetrics[param.parameter_name]) {
            const contribution = param.coefficient * adjustedMetrics[param.parameter_name];
            prediction += contribution;
            console.log(`${modelType} - ${param.parameter_name}: ${param.coefficient} * ${adjustedMetrics[param.parameter_name]} = ${contribution}`);
          }
        });

        prediction = prediction * 1.0; // 과적합 조정 제거
        predictions[modelType] = Math.max(0, Math.round(prediction));
        console.log(`${modelType} final prediction: ${predictions[modelType]}`);
      } else {
        console.log(`No regression parameters found for ${modelType}`);
      }
    });

    console.log('Final predictions:', predictions);
    setPredictedHeadcount(predictions);
  };

  const handleMetricChange = (metricName: string, value: number) => {
    const baseValue = teamMetrics[metricName] || 0;
    const adjustedValue = baseValue * (1 + value / 100);
    console.log(`Adjusting ${metricName}: base=${baseValue}, adjustment=${value}%, new=${adjustedValue}`);
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
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">조직별 Simulation (4개 모델)</h1>
        <p className="text-gray-600">조직별 인력 예측 시뮬레이션 - 전체/책임/선임/사원 모델</p>
      </div>

      {/* Miller Column Navigation */}
      <div className="grid grid-cols-6 gap-4">
        {/* 회사 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">회사</h3>
          <div className="space-y-2">
            <button
              onClick={() => setSelectedCompany('화승R&A')}
              className={`w-full px-3 py-2 text-left rounded text-sm ${
                selectedCompany === '화승R&A'
                  ? 'bg-blue-100 text-blue-800 border border-blue-300'
                  : 'hover:bg-gray-50 border border-gray-200'
              }`}
            >
              화승R&A
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
            {teams.map((team) => (
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

        {/* 분석가능팀 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">분석가능팀</h3>
          <div className="space-y-2">
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
          {/* 2025년 정원 */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600">2025년 정원</h3>
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-blue-600 text-sm font-semibold">👥</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {(currentHeadcount['총'] || 0)}명
            </div>
            <div className="text-xs text-green-600">현재 인원</div>
          </div>

          {/* 2026년 예상 정원 */}
          <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-600">2026년 예상 정원</h3>
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                <span className="text-purple-600 text-sm font-semibold">🎯</span>
              </div>
            </div>
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {(predictedHeadcount['총'] || 0)}명
            </div>
            <div className={`text-xs ${
              (predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) > 0
                ? 'text-green-600'
                : (predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) < 0
                  ? 'text-red-600'
                  : 'text-gray-600'
            }`}>
              {(predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) > 0 ? '+' : ''}
              {(predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0)}명 변화
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
              (predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) > 0
                ? 'text-green-600'
                : (predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) < 0
                  ? 'text-red-600'
                  : 'text-gray-600'
            }`}>
              {currentHeadcount['총'] > 0
                ? `${(((predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0)) / (currentHeadcount['총'] || 1) * 100).toFixed(1)}%`
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
                  return (
                    <div key={metricName} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">{metricName}</span>
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
                      <th className="text-center py-2 px-3 font-medium text-gray-700">현재</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">예측</th>
                      <th className="text-center py-2 px-3 font-medium text-gray-700">변화</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-blue-700">전체</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['총'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-blue-600 font-semibold">{predictedHeadcount['총'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0))}`}>
                        {(predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['총'] || 0) - (currentHeadcount['총'] || 0)}명
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-green-700">책임</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['책임'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-green-600 font-semibold">{predictedHeadcount['책임'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0))}`}>
                        {(predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['책임'] || 0) - (currentHeadcount['책임'] || 0)}명
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-orange-700">선임</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['선임'] || 0}명</td>
                      <td className="text-center py-2 px-3 text-orange-600 font-semibold">{predictedHeadcount['선임'] || 0}명</td>
                      <td className={`text-center py-2 px-3 font-semibold ${getChangeColor((predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0))}`}>
                        {(predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0) > 0 ? '+' : ''}
                        {(predictedHeadcount['선임'] || 0) - (currentHeadcount['선임'] || 0)}명
                      </td>
                    </tr>
                    <tr className="hover:bg-gray-50">
                      <td className="py-2 px-3 font-medium text-purple-700">사원</td>
                      <td className="text-center py-2 px-3">{currentHeadcount['사원'] || 0}명</td>
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