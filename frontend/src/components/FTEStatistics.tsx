import React, { useState, useEffect } from 'react';
import { MagicCard } from './ui/magic-card';
import { cn } from '../lib/utils';

interface FTEData {
  팀명: string;
  FTE_전체: number;
  FTE_책임: number;
  FTE_선임: number;
  FTE_사원: number;
  인원수_전체: number;
  인원수_책임: number;
  인원수_선임: number;
  인원수_사원: number;
  FTE_per_인원_전체: number;
  FTE_per_인원_책임: number;
  FTE_per_인원_선임: number;
  FTE_per_인원_사원: number;
}

interface FTEStatisticsProps {
  selectedLevel: {
    company?: string;
    department?: string;
    division?: string;
    section?: string;
    team?: string;
  };
  organizationData: any[];
  fteData: FTEData[];
  onDrillDown?: (orgName: string) => void;
}

interface MetricCardProps {
  title: string;
  value: number;
  unit: string;
  status?: 'high' | 'low' | 'normal';
  icon?: string;
  people?: number;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, unit, status, icon, people }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'high': return 'border-red-400 bg-red-50';
      case 'low': return 'border-blue-400 bg-blue-50';
      case 'normal': return 'border-green-400 bg-green-50';
      default: return 'border-gray-200 bg-white';
    }
  };

  const getIconColor = () => {
    switch (status) {
      case 'high': return 'text-red-600';
      case 'low': return 'text-blue-600';
      case 'normal': return 'text-green-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className={cn(
      "rounded-lg border-2 p-3 transition-all hover:shadow-md w-full",
      getStatusColor()
    )}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <span className={cn("text-lg", getIconColor())}>
            {icon === 'up' && '▲'}
            {icon === 'down' && '▼'}
            {icon === 'normal' && '●'}
          </span>
          <div className="text-xl font-bold text-gray-900">
            {value.toFixed(1)}
          </div>
          <span className="text-sm text-gray-600">{unit}</span>
        </div>
        {people !== undefined && people > 0 && (
          <div className="text-base font-medium text-gray-700 px-1.5 py-0.5">
            {people}명
          </div>
        )}
      </div>
    </div>
  );
};

export const FTEStatistics: React.FC<FTEStatisticsProps> = ({
  selectedLevel,
  organizationData,
  fteData,
  onDrillDown
}) => {
  const [statistics, setStatistics] = useState<any>(null);

  useEffect(() => {
    calculateStatistics();
  }, [selectedLevel, organizationData, fteData]);

  const calculateStatistics = () => {
    // 현재 선택된 레벨의 하위 조직들 찾기
    let filteredTeams: string[] = [];

    if (!selectedLevel.company) {
      // 아무것도 선택하지 않았을 때는 전체 데이터
      filteredTeams = fteData.map(f => f.팀명);
    } else {
      // 선택된 레벨에 따라 필터링
      const filtered = organizationData.filter(org => {
        if (selectedLevel.team) {
          return org.팀 === selectedLevel.team;
        }
        if (selectedLevel.section) {
          return org.회사 === selectedLevel.company &&
                 org.본부 === selectedLevel.department &&
                 org.담당_사업단_센터 === selectedLevel.division &&
                 org.실 === selectedLevel.section;
        }
        if (selectedLevel.division) {
          return org.회사 === selectedLevel.company &&
                 org.본부 === selectedLevel.department &&
                 org.담당_사업단_센터 === selectedLevel.division;
        }
        if (selectedLevel.department) {
          return org.회사 === selectedLevel.company &&
                 org.본부 === selectedLevel.department;
        }
        return org.회사 === selectedLevel.company;
      });

      filteredTeams = Array.from(new Set(filtered.map(f => f.팀)));
    }

    // FTE 데이터 필터링
    const filteredFTEData = fteData.filter(f => filteredTeams.includes(f.팀명));

    if (filteredFTEData.length === 0) {
      setStatistics(null);
      return;
    }

    // 통계 계산
    const totalPeople = filteredFTEData.reduce((sum, d) => sum + d.인원수_전체, 0);
    const totalFTE = filteredFTEData.reduce((sum, d) => sum + d.FTE_전체, 0);
    const avgFTEPerPerson = totalFTE / totalPeople;

    // 직급별 통계
    const byPosition = {
      책임: {
        fte: filteredFTEData.reduce((sum, d) => sum + d.FTE_책임, 0),
        people: filteredFTEData.reduce((sum, d) => sum + d.인원수_책임, 0),
        teams: []
      },
      선임: {
        fte: filteredFTEData.reduce((sum, d) => sum + d.FTE_선임, 0),
        people: filteredFTEData.reduce((sum, d) => sum + d.인원수_선임, 0),
        teams: []
      },
      사원: {
        fte: filteredFTEData.reduce((sum, d) => sum + d.FTE_사원, 0),
        people: filteredFTEData.reduce((sum, d) => sum + d.인원수_사원, 0),
        teams: []
      }
    };

    // 각 직급별 팀별 FTE/인원 계산
    const teamMetrics = filteredFTEData.map(team => ({
      팀명: team.팀명,
      책임: team.인원수_책임 > 0 ? team.FTE_책임 / team.인원수_책임 : 0,
      선임: team.인원수_선임 > 0 ? team.FTE_선임 / team.인원수_선임 : 0,
      사원: team.인원수_사원 > 0 ? team.FTE_사원 / team.인원수_사원 : 0
    }));

    // 하위 조직별 통계 - 중복 제거하고 그룹핑
    const groupedStats: { [key: string]: any } = {};

    filteredFTEData.forEach(team => {
      const orgInfo = organizationData.find(org => org.팀 === team.팀명);
      const displayName = getSubOrgName(orgInfo) || team.팀명;

      if (!groupedStats[displayName]) {
        groupedStats[displayName] = {
          name: displayName,
          책임: { fte: 0, people: 0, ratio: 0 },
          선임: { fte: 0, people: 0, ratio: 0 },
          사원: { fte: 0, people: 0, ratio: 0 }
        };
      }

      // FTE와 인원수 누적
      groupedStats[displayName].책임.fte += team.FTE_책임;
      groupedStats[displayName].책임.people += team.인원수_책임;
      groupedStats[displayName].선임.fte += team.FTE_선임;
      groupedStats[displayName].선임.people += team.인원수_선임;
      groupedStats[displayName].사원.fte += team.FTE_사원;
      groupedStats[displayName].사원.people += team.인원수_사원;
    });

    // 비율 계산
    Object.values(groupedStats).forEach((stat: any) => {
      stat.책임.ratio = stat.책임.people > 0 ? stat.책임.fte / stat.책임.people : 0;
      stat.선임.ratio = stat.선임.people > 0 ? stat.선임.fte / stat.선임.people : 0;
      stat.사원.ratio = stat.사원.people > 0 ? stat.사원.fte / stat.사원.people : 0;
    });

    const subOrgStats = Object.values(groupedStats).slice(0, 10);

    // 상위 20%, 하위 20% 계산
    const thresholds = calculateThresholds(teamMetrics);

    setStatistics({
      totalPeople,
      totalFTE,
      avgFTEPerPerson,
      byPosition,
      subOrgStats,
      thresholds
    });
  };

  const getSubOrgName = (orgInfo: any) => {
    if (!orgInfo) return '';

    // 현재 선택된 레벨에 따라 표시할 하위 조직명 결정
    if (!selectedLevel.company) {
      return orgInfo.회사;
    }
    if (!selectedLevel.department) {
      return orgInfo.본부;
    }
    if (!selectedLevel.division) {
      return orgInfo.담당_사업단_센터;
    }
    if (!selectedLevel.section) {
      return orgInfo.실 || orgInfo.팀;
    }
    return orgInfo.팀;
  };

  const calculateThresholds = (teamMetrics: any[]) => {
    const positions = ['책임', '선임', '사원'];
    const thresholds: any = {};

    positions.forEach(position => {
      const values = teamMetrics
        .map(t => t[position])
        .filter(v => v > 0)
        .sort((a, b) => a - b);

      if (values.length > 0) {
        const lowIndex = Math.floor(values.length * 0.2);
        const highIndex = Math.floor(values.length * 0.8);

        thresholds[position] = {
          low: values[lowIndex] || values[0],
          high: values[highIndex] || values[values.length - 1]
        };
      }
    });

    return thresholds;
  };

  const getStatus = (value: number, position: string) => {
    // 평균값에 대한 판정
    if (position === '평균') {
      if (value >= 1.4) return 'high';
      if (value <= 0.9) return 'low';
      return 'normal';
    }

    // 직급별 판정
    if (!statistics || !statistics.thresholds[position]) return 'normal';

    const threshold = statistics.thresholds[position];
    if (value >= threshold.high) return 'high';
    if (value <= threshold.low) return 'low';
    return 'normal';
  };

  const getIcon = (status: string) => {
    switch (status) {
      case 'high': return 'up';
      case 'low': return 'down';
      default: return 'normal';
    }
  };

  if (!statistics) {
    return (
      <div className="text-center py-8 text-gray-500">
        데이터를 불러오는 중...
      </div>
    );
  }

  return (
    <div className="space-y-6 mt-6">
      {/* 전체 요약 */}
      <MagicCard className="p-6 bg-white">
        <h3 className="text-lg font-semibold mb-4">인력 수준 적정성 분석</h3>
        <p className="text-sm text-gray-600 mb-4">
          현재 조직별/직급별 인력 수준 대비 근무기록 기반 FTE
        </p>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="border-2 border-gray-300 bg-white rounded-lg p-4 shadow-md">
            <div className="text-3xl font-bold text-blue-600">
              {statistics.totalPeople}명
            </div>
            <div className="text-sm text-gray-600">총 분석 인원</div>
          </div>
          <div className="border-2 border-gray-300 bg-white rounded-lg p-4 shadow-md">
            <div className="text-3xl font-bold text-blue-600">
              {statistics.avgFTEPerPerson.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">조직별 FTE/인력 평균</div>
          </div>
        </div>

        {/* 전체 현황 테이블 - 조직이 가로, 직급이 세로 */}
        <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-sm text-gray-600 border-b-2 border-gray-300">
                <th className="text-left py-3 px-3 font-medium">구분</th>
                <th className="border-l-2 border-gray-300"></th>
                {statistics.subOrgStats.map((org: any, idx: number) => (
                  <th
                    key={idx}
                    className={cn(
                      "text-center py-3 px-2 min-w-[160px] text-base font-semibold",
                      onDrillDown && "cursor-pointer hover:bg-gray-100 hover:text-blue-600 transition-colors"
                    )}
                    onClick={() => onDrillDown && onDrillDown(org.name)}
                  >
                    {org.name}
                  </th>
                ))}
                <th className="border-l-2 border-gray-300"></th>
                <th className="text-center py-3 px-2 min-w-[160px] text-base font-semibold bg-gray-100">전체 평균</th>
              </tr>
            </thead>
            <tbody>
              {/* 조직별 평균 행 */}
              <tr className="border-b-2 border-gray-300 bg-blue-50">
                <td className="py-3 px-3 font-medium text-gray-700">조직별 평균</td>
                <td className="border-l-2 border-gray-300"></td>
                {statistics.subOrgStats.map((org: any, idx: number) => {
                  const totalPeople = org.책임.people + org.선임.people + org.사원.people;
                  const totalFTE = org.책임.fte + org.선임.fte + org.사원.fte;
                  const avgRatio = totalPeople > 0 ? totalFTE / totalPeople : 0;

                  return (
                    <td key={idx} className="text-center p-3">
                      <MetricCard
                        title="평균"
                        value={totalFTE}
                        unit="FTE"
                        status={getStatus(avgRatio, '평균')}
                        icon={getIcon(getStatus(avgRatio, '평균'))}
                        people={totalPeople}
                      />
                    </td>
                  );
                })}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <div className="rounded-lg border-2 border-purple-400 bg-purple-50 p-3 min-w-[120px]">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1">
                        <span className="text-lg text-purple-600">★</span>
                        <div className="text-xl font-bold text-gray-900">
                          {statistics.totalFTE.toFixed(1)}
                        </div>
                        <span className="text-sm text-gray-600">FTE</span>
                      </div>
                      <div className="text-base font-medium text-gray-700 px-1.5 py-0.5">
                        {statistics.totalPeople}명
                      </div>
                    </div>
                  </div>
                </td>
              </tr>

              {/* 구분선 */}
              <tr>
                <td colSpan={statistics.subOrgStats.length + 4} className="p-0"></td>
              </tr>

              {/* 책임 행 */}
              <tr className="border-b border-gray-200">
                <td className="py-3 px-3 font-medium text-gray-700">책임</td>
                <td className="border-l-2 border-gray-300"></td>
                {statistics.subOrgStats.map((org: any, idx: number) => (
                  <td key={idx} className="text-center p-3">
                    <MetricCard
                      title="책임"
                      value={org.책임.fte}
                      unit="FTE"
                      status={getStatus(org.책임.ratio, '책임')}
                      icon={getIcon(getStatus(org.책임.ratio, '책임'))}
                      people={org.책임.people}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <div className="rounded-lg border-2 border-blue-400 bg-blue-50 p-3 min-w-[120px]">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1">
                        <span className="text-lg text-blue-600">●</span>
                        <div className="text-xl font-bold text-gray-900">
                          {statistics.byPosition.책임.fte.toFixed(1)}
                        </div>
                        <span className="text-sm text-gray-600">FTE</span>
                      </div>
                      <div className="text-base font-medium text-gray-700 px-1.5 py-0.5">
                        {statistics.byPosition.책임.people}명
                      </div>
                    </div>
                  </div>
                </td>
              </tr>

              {/* 선임 행 */}
              <tr className="border-b border-gray-200">
                <td className="py-3 px-3 font-medium text-gray-700">선임</td>
                <td className="border-l-2 border-gray-300"></td>
                {statistics.subOrgStats.map((org: any, idx: number) => (
                  <td key={idx} className="text-center p-3">
                    <MetricCard
                      title="선임"
                      value={org.선임.fte}
                      unit="FTE"
                      status={getStatus(org.선임.ratio, '선임')}
                      icon={getIcon(getStatus(org.선임.ratio, '선임'))}
                      people={org.선임.people}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <div className="rounded-lg border-2 border-blue-400 bg-blue-50 p-3 min-w-[120px]">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1">
                        <span className="text-lg text-blue-600">●</span>
                        <div className="text-xl font-bold text-gray-900">
                          {statistics.byPosition.선임.fte.toFixed(1)}
                        </div>
                        <span className="text-sm text-gray-600">FTE</span>
                      </div>
                      <div className="text-base font-medium text-gray-700 px-1.5 py-0.5">
                        {statistics.byPosition.선임.people}명
                      </div>
                    </div>
                  </div>
                </td>
              </tr>

              {/* 사원 행 */}
              <tr className="border-b border-gray-200">
                <td className="py-3 px-3 font-medium text-gray-700">사원</td>
                <td className="border-l-2 border-gray-300"></td>
                {statistics.subOrgStats.map((org: any, idx: number) => (
                  <td key={idx} className="text-center p-3">
                    <MetricCard
                      title="사원"
                      value={org.사원.fte}
                      unit="FTE"
                      status={getStatus(org.사원.ratio, '사원')}
                      icon={getIcon(getStatus(org.사원.ratio, '사원'))}
                      people={org.사원.people}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <div className="rounded-lg border-2 border-blue-400 bg-blue-50 p-3 min-w-[120px]">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1">
                        <span className="text-lg text-blue-600">●</span>
                        <div className="text-xl font-bold text-gray-900">
                          {statistics.byPosition.사원.fte.toFixed(1)}
                        </div>
                        <span className="text-sm text-gray-600">FTE</span>
                      </div>
                      <div className="text-base font-medium text-gray-700 px-1.5 py-0.5">
                        {statistics.byPosition.사원.people}명
                      </div>
                    </div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* FTE 설명 */}
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mt-6 mb-6">
          <p className="text-sm text-gray-700">
            <span className="font-semibold">FTE (Full-Time Equivalent)</span>: 정규직 환산 인원으로, 실제 근무시간을 정규직 기준 근무시간으로 나눈 값입니다.
            예) FTE 1.2 = 정규직 대비 120% 근무, FTE 0.8 = 정규직 대비 80% 근무
          </p>
        </div>

        {/* 구분선 */}
        <div className="mb-6 border-t-2 border-gray-300"></div>

        {/* 범례 */}
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border-2 border-red-400 bg-red-50 px-6 py-4 shadow-md">
            <div className="flex items-center gap-3">
              <span className="text-red-600 text-3xl">▲</span>
              <div className="flex-1">
                <div className="font-semibold text-lg text-gray-900">과부하 가능성 존재</div>
                <div className="text-sm text-gray-600">상위 20% (≥1.4)</div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border-2 border-green-400 bg-green-50 px-6 py-4 shadow-md">
            <div className="flex items-center gap-3">
              <span className="text-green-600 text-3xl">●</span>
              <div className="flex-1">
                <div className="font-semibold text-lg text-gray-900">적정 인력 수준</div>
                <div className="text-sm text-gray-600">중간 60% (0.9~1.3)</div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border-2 border-blue-400 bg-blue-50 px-6 py-4 shadow-md">
            <div className="flex items-center gap-3">
              <span className="text-blue-600 text-3xl">▼</span>
              <div className="flex-1">
                <div className="font-semibold text-lg text-gray-900">인력 과다 가능성 존재</div>
                <div className="text-sm text-gray-600">하위 20% (≤0.9)</div>
              </div>
            </div>
          </div>
        </div>
      </MagicCard>
    </div>
  );
};