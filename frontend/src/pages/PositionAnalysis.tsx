import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import initSqlJs from 'sql.js';

interface TeamPrediction {
  team_name: string;
  position: string;
  current_headcount: number;
  predicted_headcount: number;
  change: number;
  change_percent: number;
  category: string;
}

interface TeamData {
  team_name: string;
  책임: { predicted: number; change: number; category: string };
  선임: { predicted: number; change: number; category: string };
  사원: { predicted: number; change: number; category: string };
}

const PositionAnalysis: React.FC = () => {
  const [teamsData, setTeamsData] = useState<TeamData[]>([]);
  const [summary, setSummary] = useState<{
    totalCurrent: number;
    totalPredicted: number;
    책임Change: number;
    선임Change: number;
    사원Change: number;
    책임Total: number;
    선임Total: number;
    사원Total: number;
  }>({
    totalCurrent: 0,
    totalPredicted: 0,
    책임Change: 0,
    선임Change: 0,
    사원Change: 0,
    책임Total: 0,
    선임Total: 0,
    사원Total: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    try {
      setLoading(true);

      // sql.js 초기화
      const SQL = await initSqlJs({
        locateFile: (file: string) => `https://sql.js.org/dist/${file}`
      });

      const response = await fetch(`/hwaseung_RnD.db?t=${Date.now()}`);
      const buffer = await response.arrayBuffer();
      const db = new SQL.Database(new Uint8Array(buffer));

      // team_predictions 테이블에서 모든 데이터 조회
      const result = db.exec(`
        SELECT team_name, position, current_headcount, predicted_headcount, change, change_percent, category
        FROM team_predictions
        ORDER BY team_name, CASE position
          WHEN '총합' THEN 1
          WHEN '책임' THEN 2
          WHEN '선임' THEN 3
          WHEN '사원' THEN 4
        END
      `);

      if (result.length > 0) {
        const predictions: TeamPrediction[] = result[0].values.map(row => ({
          team_name: row[0] as string,
          position: row[1] as string,
          current_headcount: row[2] as number,
          predicted_headcount: row[3] as number,
          change: row[4] as number,
          change_percent: row[5] as number,
          category: row[6] as string
        }));

        // 팀별로 데이터 그룹핑
        const teams: { [key: string]: TeamData } = {};
        predictions.forEach(p => {
          if (!teams[p.team_name]) {
            teams[p.team_name] = {
              team_name: p.team_name,
              책임: { predicted: 0, change: 0, category: '적정' },
              선임: { predicted: 0, change: 0, category: '적정' },
              사원: { predicted: 0, change: 0, category: '적정' }
            };
          }

          if (p.position === '책임' || p.position === '선임' || p.position === '사원') {
            teams[p.team_name][p.position] = {
              predicted: p.predicted_headcount,
              change: p.change,
              category: p.category
            };
          }
        });

        const teamsArray = Object.values(teams);
        setTeamsData(teamsArray);

        // 요약 통계 계산
        let totalCurrent = 0;
        let totalPredicted = 0;
        let 책임Change = 0;
        let 선임Change = 0;
        let 사원Change = 0;
        let 책임Total = 0;
        let 선임Total = 0;
        let 사원Total = 0;

        predictions.forEach(p => {
          if (p.position === '총') {
            totalCurrent += p.current_headcount;
          }
          if (p.position === '책임') {
            totalPredicted += p.predicted_headcount;
            책임Change += p.change;
            책임Total += p.predicted_headcount;
          }
          if (p.position === '선임') {
            totalPredicted += p.predicted_headcount;
            선임Change += p.change;
            선임Total += p.predicted_headcount;
          }
          if (p.position === '사원') {
            totalPredicted += p.predicted_headcount;
            사원Change += p.change;
            사원Total += p.predicted_headcount;
          }
        });

        setSummary({
          totalCurrent,
          totalPredicted,
          책임Change,
          선임Change,
          사원Change,
          책임Total,
          선임Total,
          사원Total
        });

        console.log('✅ 예측 데이터 로드 완료:', teamsArray);
      }

      db.close();
      setLoading(false);
    } catch (error) {
      console.error('❌ 예측 데이터 로드 실패:', error);
      setLoading(false);
    }
  };

  const getCellStyle = (category: string) => {
    switch (category) {
      case '충원필요':
        return 'bg-red-200 text-red-800 border-red-200';
      case '감원검토':
        return 'bg-blue-200 text-blue-800 border-blue-200';
      case '적정':
      default:
        return 'bg-green-200 text-green-800 border-green-200';
    }
  };

  const getIcon = (category: string) => {
    switch (category) {
      case '충원필요':
        return '▲';
      case '감원검토':
        return '▼';
      case '적정':
      default:
        return '●';
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-foreground">조직, 직급별 적정인력 산정</h1>
            <p className="text-muted-foreground">조직, 직급별 적정인력 산정 요약</p>
          </div>
        </div>
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">데이터 로딩 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">조직, 직급별 적정인력 산정</h1>
          <p className="text-muted-foreground">조직, 직급별 적정인력 산정 요약</p>
        </div>
      </div>

      {/* 통계 카드들 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="rounded-lg border-2 border-gray-300 bg-gray-50 p-4 text-center">
          <div className="text-2xl font-bold text-black mb-2">{summary.totalPredicted}</div>
          <div className="text-sm text-gray-600">총 분석 인원</div>
          <div className="text-xs text-gray-500 mt-1">(예상 인력 합계)</div>
        </div>
        <div className={`rounded-lg border-2 border-gray-300 p-4 text-center ${
          summary.책임Change > 0
            ? 'bg-red-50'
            : summary.책임Change < 0
              ? 'bg-blue-50'
              : 'bg-green-50'
        }`}>
          <div className={`text-2xl font-bold mb-2 ${summary.책임Change > 0 ? 'text-red-600' : summary.책임Change < 0 ? 'text-blue-600' : 'text-green-600'}`}>
            {summary.책임Change > 0 ? '+' : ''}{summary.책임Change}명
          </div>
          <div className="text-sm text-gray-600">책임 변동</div>
        </div>
        <div className={`rounded-lg border-2 border-gray-300 p-4 text-center ${
          summary.선임Change > 0
            ? 'bg-red-50'
            : summary.선임Change < 0
              ? 'bg-blue-50'
              : 'bg-green-50'
        }`}>
          <div className={`text-2xl font-bold mb-2 ${summary.선임Change > 0 ? 'text-red-600' : summary.선임Change < 0 ? 'text-blue-600' : 'text-green-600'}`}>
            {summary.선임Change > 0 ? '+' : ''}{summary.선임Change}명
          </div>
          <div className="text-sm text-gray-600">선임 변동</div>
        </div>
        <div className={`rounded-lg border-2 border-gray-300 p-4 text-center ${
          summary.사원Change > 0
            ? 'bg-red-50'
            : summary.사원Change < 0
              ? 'bg-blue-50'
              : 'bg-green-50'
        }`}>
          <div className={`text-2xl font-bold mb-2 ${summary.사원Change > 0 ? 'text-red-600' : summary.사원Change < 0 ? 'text-blue-600' : 'text-green-600'}`}>
            {summary.사원Change > 0 ? '+' : ''}{summary.사원Change}명
          </div>
          <div className="text-sm text-gray-600">사원 변동</div>
        </div>
      </div>

      {/* 직급별 적정인원(예측) 테이블 카드 */}
      <Card className="border-2 border-gray-300 bg-white">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-bold">직급별 적정인원(예측)</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
            {/* 헤더 */}
            <div className="flex border-b min-w-max">
              <div className="py-3 px-4 font-bold text-gray-900 text-base text-center flex items-center justify-center w-40 flex-shrink-0 bg-white sticky left-0 z-10">구분</div>
              {teamsData.map((team, idx) => (
                <div key={idx} className="py-3 px-4 font-bold text-gray-900 text-base text-center border-l w-64 flex-shrink-0">
                  {team.team_name}
                </div>
              ))}
            </div>

            {/* 전체 합계 행 */}
            <div className="flex border-b bg-blue-50 min-w-max">
              <div className="py-4 px-4 font-bold text-gray-900 text-base text-center flex items-center justify-center w-40 flex-shrink-0 bg-blue-50 sticky left-0 z-10">전체</div>
              {teamsData.map((team, idx) => {
                const totalChange = team.책임.change + team.선임.change + team.사원.change;
                const getBgColor = () => {
                  if (totalChange > 0) return 'bg-red-50';
                  if (totalChange < 0) return 'bg-blue-50';
                  return 'bg-green-50';
                };
                const getBorderColor = () => {
                  if (totalChange > 0) return 'border-red-500';
                  if (totalChange < 0) return 'border-blue-500';
                  return 'border-green-500';
                };
                const getIconColor = () => {
                  if (totalChange > 0) return 'text-red-700';
                  if (totalChange < 0) return 'text-blue-700';
                  return 'text-green-700';
                };
                const getTextColor = () => {
                  if (totalChange > 0) return 'text-red-900';
                  if (totalChange < 0) return 'text-blue-900';
                  return 'text-green-900';
                };
                const getIcon = () => {
                  if (totalChange > 0) return '▲';
                  if (totalChange < 0) return '▼';
                  return '●';
                };
                return (
                  <div key={idx} className="py-4 px-4 text-center border-l flex items-center justify-center w-64 flex-shrink-0">
                    <div style={{width: '200px'}} className={`flex items-center justify-center gap-1.5 px-4 py-3 rounded-lg border-2 ${getBgColor()} ${getBorderColor()}`}>
                      <span className={`text-base ${getIconColor()}`}>{getIcon()}</span>
                      <span className={`text-base font-bold ${getTextColor()}`}>
                        {team.책임.predicted + team.선임.predicted + team.사원.predicted}명
                      </span>
                      {totalChange !== 0 && (
                        <span className={`text-base ${getIconColor()}`}>
                          ({totalChange > 0 ? '+' : ''}{totalChange})
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 책임 행 */}
            <div className="flex border-b min-w-max">
              <div className="py-4 px-4 font-bold text-gray-900 text-base text-center flex items-center justify-center w-40 flex-shrink-0 bg-white sticky left-0 z-10">책임</div>
              {teamsData.map((team, idx) => (
                <div key={idx} className="py-4 px-4 text-center border-l flex items-center justify-center w-64 flex-shrink-0">
                  <div style={{width: '200px'}} className={`flex items-center justify-center gap-1.5 px-4 py-3 rounded-lg border-2 ${
                    team.책임.category === '충원필요' ? 'bg-red-50 border-red-500' :
                    team.책임.category === '감원검토' ? 'bg-blue-50 border-blue-500' :
                    'bg-green-50 border-green-500'
                  }`}>
                    <span className={`text-base ${
                      team.책임.category === '충원필요' ? 'text-red-700' :
                      team.책임.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {getIcon(team.책임.category)}
                    </span>
                    <span className={`text-base font-bold ${
                      team.책임.category === '충원필요' ? 'text-red-700' :
                      team.책임.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {team.책임.predicted}명
                    </span>
                    {team.책임.change !== 0 && (
                      <span className={`text-base ${
                        team.책임.category === '충원필요' ? 'text-red-600' :
                        team.책임.category === '감원검토' ? 'text-blue-600' :
                        'text-green-600'
                      }`}>
                        ({team.책임.change > 0 ? '+' : ''}{team.책임.change})
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* 선임 행 */}
            <div className="flex border-b min-w-max">
              <div className="py-4 px-4 font-bold text-gray-900 text-base text-center flex items-center justify-center w-40 flex-shrink-0 bg-white sticky left-0 z-10">선임</div>
              {teamsData.map((team, idx) => (
                <div key={idx} className="py-4 px-4 text-center border-l flex items-center justify-center w-64 flex-shrink-0">
                  <div style={{width: '200px'}} className={`flex items-center justify-center gap-1.5 px-4 py-3 rounded-lg border-2 ${
                    team.선임.category === '충원필요' ? 'bg-red-50 border-red-500' :
                    team.선임.category === '감원검토' ? 'bg-blue-50 border-blue-500' :
                    'bg-green-50 border-green-500'
                  }`}>
                    <span className={`text-base ${
                      team.선임.category === '충원필요' ? 'text-red-700' :
                      team.선임.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {getIcon(team.선임.category)}
                    </span>
                    <span className={`text-base font-bold ${
                      team.선임.category === '충원필요' ? 'text-red-700' :
                      team.선임.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {team.선임.predicted}명
                    </span>
                    {team.선임.change !== 0 && (
                      <span className={`text-base ${
                        team.선임.category === '충원필요' ? 'text-red-600' :
                        team.선임.category === '감원검토' ? 'text-blue-600' :
                        'text-green-600'
                      }`}>
                        ({team.선임.change > 0 ? '+' : ''}{team.선임.change})
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* 사원 행 */}
            <div className="flex min-w-max">
              <div className="py-4 px-4 font-bold text-gray-900 text-base text-center flex items-center justify-center w-40 flex-shrink-0 bg-white sticky left-0 z-10">사원</div>
              {teamsData.map((team, idx) => (
                <div key={idx} className="py-4 px-4 text-center border-l flex items-center justify-center w-64 flex-shrink-0">
                  <div style={{width: '200px'}} className={`flex items-center justify-center gap-1.5 px-4 py-3 rounded-lg border-2 ${
                    team.사원.category === '충원필요' ? 'bg-red-50 border-red-500' :
                    team.사원.category === '감원검토' ? 'bg-blue-50 border-blue-500' :
                    'bg-green-50 border-green-500'
                  }`}>
                    <span className={`text-base ${
                      team.사원.category === '충원필요' ? 'text-red-700' :
                      team.사원.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {getIcon(team.사원.category)}
                    </span>
                    <span className={`text-base font-bold ${
                      team.사원.category === '충원필요' ? 'text-red-700' :
                      team.사원.category === '감원검토' ? 'text-blue-700' :
                      'text-green-700'
                    }`}>
                      {team.사원.predicted}명
                    </span>
                    {team.사원.change !== 0 && (
                      <span className={`text-base ${
                        team.사원.category === '충원필요' ? 'text-red-600' :
                        team.사원.category === '감원검토' ? 'text-blue-600' :
                        'text-green-600'
                      }`}>
                        ({team.사원.change > 0 ? '+' : ''}{team.사원.change})
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 범례 */}
          <div className="mt-6 pt-4 border-t flex items-center gap-6 text-sm text-muted-foreground px-4 pb-4">
            <div className="flex items-center gap-2">
              <span className="text-red-600 text-xl">▲</span>
              <span>충원필요</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-green-600 text-xl">●</span>
              <span>적정</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-600 text-xl">▼</span>
              <span>감원검토</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PositionAnalysis;
