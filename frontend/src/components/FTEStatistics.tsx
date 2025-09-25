import React, { useState, useEffect } from 'react';
import { MagicCard } from './ui/magic-card';
import { cn } from '../lib/utils';

interface FTEData {
  팀명: string;
  회사: string;
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
  fteValue?: number;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, unit, status, icon, people, fteValue }) => {
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
      "rounded-lg border-2 p-3 transition-all hover:shadow-md",
      "min-w-[140px] w-full", // 최소 너비 설정
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
        <div className="text-right min-w-[60px]"> {/* 오른쪽 영역 최소 너비 */}
          {people !== undefined && people > 0 && (
            <div className="text-base font-medium text-gray-700">
              {people}명
            </div>
          )}
          {fteValue !== undefined && (
            <div className="text-xs text-gray-500 whitespace-nowrap">
              ({fteValue.toFixed(1)} FTE)
            </div>
          )}
        </div>
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
    let isCompanyGrouped = false;

    if (!selectedLevel.company) {
      // 아무것도 선택하지 않았을 때는 전체 데이터를 회사별로 표시
      filteredTeams = fteData.map(f => f.팀명);
      isCompanyGrouped = true;
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

    // FTE 데이터 필터링 - 팀명과 회사를 모두 확인
    const filteredFTEData = fteData.filter(f => {
      if (!filteredTeams.includes(f.팀명)) {
        return false;
      }

      // 회사별로 필터링: organization 테이블에서 해당 팀의 회사와 FTE의 회사가 일치해야 함
      const orgInfo = organizationData.find(org =>
        org.팀 === f.팀명 &&
        (!selectedLevel.company || org.회사 === selectedLevel.company)
      );

      if (!orgInfo) {
        return false;
      }

      // FTE 데이터의 회사와 organization 데이터의 회사가 일치해야 함
      return f.회사 === orgInfo.회사;
    });

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

    if (isCompanyGrouped) {
      // 회사별로 그룹핑 - organization 테이블에 있는 팀만 처리
      filteredFTEData.forEach(team => {
        const orgInfo = organizationData.find(org => org.팀 === team.팀명);

        // organization 테이블에 없는 팀은 제외
        if (!orgInfo) {
          return;
        }

        const companyName = orgInfo.회사;

        if (!groupedStats[companyName]) {
          groupedStats[companyName] = {
            name: companyName,
            책임: { fte: 0, people: 0, ratio: 0 },
            선임: { fte: 0, people: 0, ratio: 0 },
            사원: { fte: 0, people: 0, ratio: 0 }
          };
        }

        // FTE와 인원수 누적
        groupedStats[companyName].책임.fte += team.FTE_책임;
        groupedStats[companyName].책임.people += team.인원수_책임;
        groupedStats[companyName].선임.fte += team.FTE_선임;
        groupedStats[companyName].선임.people += team.인원수_선임;
        groupedStats[companyName].사원.fte += team.FTE_사원;
        groupedStats[companyName].사원.people += team.인원수_사원;
      });
    } else {
      // 기존 로직: 선택된 레벨에 따라 그룹핑
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
    }

    // 비율 계산
    Object.values(groupedStats).forEach((stat: any) => {
      stat.책임.ratio = stat.책임.people > 0 ? stat.책임.fte / stat.책임.people : 0;
      stat.선임.ratio = stat.선임.people > 0 ? stat.선임.fte / stat.선임.people : 0;
      stat.사원.ratio = stat.사원.people > 0 ? stat.사원.fte / stat.사원.people : 0;
    });

    const subOrgStats = Object.values(groupedStats).slice(0, 10);

    // 상위 20%, 하위 20% 계산 - subOrgStats 기준으로
    const thresholds = calculateThresholds(subOrgStats);

    // 디버깅용 로그
    console.log('=== Unified Threshold Calculation ===');
    console.log('Single threshold for all values:', thresholds);
    console.log('SubOrgStats values:', subOrgStats.map(org => ({
      name: org.name,
      avgRatio: ((org.책임.fte + org.선임.fte + org.사원.fte) / (org.책임.people + org.선임.people + org.사원.people)).toFixed(2),
      책임: org.책임.ratio.toFixed(2),
      선임: org.선임.ratio.toFixed(2),
      사원: org.사원.ratio.toFixed(2)
    })));

    // 각 값에 대한 판정 결과 확인 - 이제 모든 position에서 동일해야 함
    const testValues = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    console.log('Test status for common values (should be same across all positions):');
    testValues.forEach(val => {
      const status = getStatus(val, 'any'); // position 파라미터는 더 이상 사용되지 않음
      console.log(`Value ${val}: ${status}`);
    });

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

  const calculateThresholds = (subOrgStats: any[]) => {
    // 화면에 보이는 모든 카드의 값을 수집 (조직별 평균 + 모든 직급별 값)
    let allValues: number[] = [];

    subOrgStats.forEach(org => {
      // 조직별 평균 추가
      const totalPeople = org.책임.people + org.선임.people + org.사원.people;
      const totalFTE = org.책임.fte + org.선임.fte + org.사원.fte;
      if (totalPeople > 0) {
        const avgRatio = totalFTE / totalPeople;
        allValues.push(avgRatio);
      }

      // 각 직급별 값 추가
      if (org.책임.people > 0) {
        allValues.push(org.책임.ratio);
      }
      if (org.선임.people > 0) {
        allValues.push(org.선임.ratio);
      }
      if (org.사원.people > 0) {
        allValues.push(org.사원.ratio);
      }
    });

    // 값이 있는 것들만 필터링하고 정렬
    allValues = allValues.filter(v => v > 0);
    allValues.sort((a, b) => a - b);

    if (allValues.length > 0) {
      // 정확한 20%, 80% 퍼센타일 계산
      const lowIndex = Math.floor(allValues.length * 0.2);
      const highIndex = Math.floor(allValues.length * 0.8);

      // 하위 20% = 20 퍼센타일 이하
      // 상위 20% = 80 퍼센타일 이상
      return {
        low: allValues[Math.min(lowIndex, allValues.length - 1)],
        high: allValues[Math.min(highIndex, allValues.length - 1)]
      };
    } else {
      // 기본값 설정
      return {
        low: 0.9,
        high: 1.3
      };
    }
  };

  const getStatus = (value: number, position: string) => {
    // 모든 값에 대해 통합된 임계값 사용
    if (!statistics || !statistics.thresholds) {
      // 기본값 사용
      if (value >= 1.3) return 'high';
      if (value <= 1.0) return 'low';
      return 'normal';
    }

    // 상위 20% 이상
    if (value >= statistics.thresholds.high) return 'high';
    // 하위 20% 이하
    if (value <= statistics.thresholds.low) return 'low';
    // 중간 60%
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
          <table className="w-full table-fixed">
            <colgroup>
              <col className="w-24" />
              <col className="w-2" />
              {statistics.subOrgStats.map((_: any, idx: number) => (
                <col key={idx} className="w-44" />
              ))}
              <col className="w-2" />
              <col className="w-44" />
            </colgroup>
            <thead>
              <tr className="text-sm text-gray-600 border-b-2 border-gray-300">
                <th className="text-left py-3 px-3 font-medium">구분</th>
                <th className="border-l-2 border-gray-300"></th>
                {statistics.subOrgStats.map((org: any, idx: number) => (
                  <th
                    key={idx}
                    className={cn(
                      "text-center py-3 px-2 text-base font-semibold",
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
                        value={avgRatio}
                        unit=""
                        status={getStatus(avgRatio, '평균')}
                        icon={getIcon(getStatus(avgRatio, '평균'))}
                        people={totalPeople}
                        fteValue={totalFTE}
                      />
                    </td>
                  );
                })}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <MetricCard
                    title="전체평균"
                    value={statistics.avgFTEPerPerson}
                    unit=""
                    status={getStatus(statistics.avgFTEPerPerson, '평균')}
                    icon={getIcon(getStatus(statistics.avgFTEPerPerson, '평균'))}
                    people={statistics.totalPeople}
                    fteValue={statistics.totalFTE}
                  />
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
                      value={org.책임.ratio}
                      unit=""
                      status={getStatus(org.책임.ratio, '책임')}
                      icon={getIcon(getStatus(org.책임.ratio, '책임'))}
                      people={org.책임.people}
                      fteValue={org.책임.fte}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <MetricCard
                    title="책임평균"
                    value={statistics.byPosition.책임.people > 0
                      ? statistics.byPosition.책임.fte / statistics.byPosition.책임.people
                      : 0}
                    unit=""
                    status={getStatus(
                      statistics.byPosition.책임.people > 0
                        ? statistics.byPosition.책임.fte / statistics.byPosition.책임.people
                        : 0, '책임'
                    )}
                    icon={getIcon(getStatus(
                      statistics.byPosition.책임.people > 0
                        ? statistics.byPosition.책임.fte / statistics.byPosition.책임.people
                        : 0, '책임'
                    ))}
                    people={statistics.byPosition.책임.people}
                    fteValue={statistics.byPosition.책임.fte}
                  />
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
                      value={org.선임.ratio}
                      unit=""
                      status={getStatus(org.선임.ratio, '선임')}
                      icon={getIcon(getStatus(org.선임.ratio, '선임'))}
                      people={org.선임.people}
                      fteValue={org.선임.fte}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <MetricCard
                    title="선임평균"
                    value={statistics.byPosition.선임.people > 0
                      ? statistics.byPosition.선임.fte / statistics.byPosition.선임.people
                      : 0}
                    unit=""
                    status={getStatus(
                      statistics.byPosition.선임.people > 0
                        ? statistics.byPosition.선임.fte / statistics.byPosition.선임.people
                        : 0, '선임'
                    )}
                    icon={getIcon(getStatus(
                      statistics.byPosition.선임.people > 0
                        ? statistics.byPosition.선임.fte / statistics.byPosition.선임.people
                        : 0, '선임'
                    ))}
                    people={statistics.byPosition.선임.people}
                    fteValue={statistics.byPosition.선임.fte}
                  />
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
                      value={org.사원.ratio}
                      unit=""
                      status={getStatus(org.사원.ratio, '사원')}
                      icon={getIcon(getStatus(org.사원.ratio, '사원'))}
                      people={org.사원.people}
                      fteValue={org.사원.fte}
                    />
                  </td>
                ))}
                <td className="border-l-2 border-gray-300"></td>
                <td className="text-center p-3 bg-gray-50">
                  <MetricCard
                    title="사원평균"
                    value={statistics.byPosition.사원.people > 0
                      ? statistics.byPosition.사원.fte / statistics.byPosition.사원.people
                      : 0}
                    unit=""
                    status={getStatus(
                      statistics.byPosition.사원.people > 0
                        ? statistics.byPosition.사원.fte / statistics.byPosition.사원.people
                        : 0, '사원'
                    )}
                    icon={getIcon(getStatus(
                      statistics.byPosition.사원.people > 0
                        ? statistics.byPosition.사원.fte / statistics.byPosition.사원.people
                        : 0, '사원'
                    ))}
                    people={statistics.byPosition.사원.people}
                    fteValue={statistics.byPosition.사원.fte}
                  />
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
                <div className="text-sm text-gray-600">
                  상위 20% (현재 화면 기준 {statistics.thresholds?.high ? `≥${statistics.thresholds.high.toFixed(2)}` : '≥1.3'})
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border-2 border-green-400 bg-green-50 px-6 py-4 shadow-md">
            <div className="flex items-center gap-3">
              <span className="text-green-600 text-3xl">●</span>
              <div className="flex-1">
                <div className="font-semibold text-lg text-gray-900">적정 인력 수준</div>
                <div className="text-sm text-gray-600">
                  중간 60% (현재 화면 기준 {statistics.thresholds ?
                    `${(statistics.thresholds.low + 0.01).toFixed(2)}~${(statistics.thresholds.high - 0.01).toFixed(2)}` : '1.01~1.29'})
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border-2 border-blue-400 bg-blue-50 px-6 py-4 shadow-md">
            <div className="flex items-center gap-3">
              <span className="text-blue-600 text-3xl">▼</span>
              <div className="flex-1">
                <div className="font-semibold text-lg text-gray-900">인력 과다 가능성 존재</div>
                <div className="text-sm text-gray-600">
                  하위 20% (현재 화면 기준 {statistics.thresholds?.low ? `≤${statistics.thresholds.low.toFixed(2)}` : '≤1.0'})
                </div>
              </div>
            </div>
          </div>
        </div>
      </MagicCard>
    </div>
  );
};