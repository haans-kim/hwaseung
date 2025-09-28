import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { ChevronRight, Home } from 'lucide-react';
import initSqlJs from 'sql.js';
import { FTEStatistics } from '../components/FTEStatistics';

interface OrganizationNode {
  회사: string;
  본부: string;
  담당_사업단_센터: string;
  실: string;
  팀: string;
}

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

interface BreadcrumbItem {
  label: string;
  value: string;
}

// ColumnPane 컴포넌트
const ColumnPane = ({
  title,
  items,
  selectedValue,
  onSelect,
  width,
}: {
  title: string;
  items: string[];
  selectedValue: string | null;
  onSelect: (value: string) => void;
  width: string;
}) => {
  return (
    <div
      className="border-r border-gray-200 last:border-r-0 flex-shrink-0"
      style={{ width }}
    >
      <div className="bg-white px-4 py-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
      </div>
      <div className="overflow-y-auto h-[300px]">
        {items.length === 0 ? (
          <div className="px-4 py-3 text-sm text-gray-500">데이터 없음</div>
        ) : (
          items.map((item) => {
            const isSelected = selectedValue === item;
            return (
              <div
                key={item}
                onClick={() => onSelect(item)}
                className={`
                  px-4 py-2 cursor-pointer text-sm
                  ${
                    isSelected
                      ? 'bg-blue-50 text-blue-700 font-medium'
                      : 'hover:bg-gray-50 text-gray-900'
                  }
                `}
              >
                {item}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

// Breadcrumb 컴포넌트
const Breadcrumb = ({
  items,
  onNavigate
}: {
  items: BreadcrumbItem[];
  onNavigate: (level: string) => void;
}) => {
  return (
    <nav className="flex items-center space-x-1 text-sm mb-4">
      <div
        className="flex items-center hover:text-blue-600 transition-colors cursor-pointer"
        onClick={() => onNavigate('home')}
      >
        <Home className="w-4 h-4" />
      </div>

      {items.map((item, index) => (
        <div key={index} className="flex items-center">
          <ChevronRight className="w-4 h-4 mx-1 text-gray-400" />
          <span
            className="text-gray-900 font-medium hover:text-blue-600 transition-colors cursor-pointer"
            onClick={() => onNavigate(item.value)}
          >
            {item.label}
          </span>
        </div>
      ))}
    </nav>
  );
};

const OrganizationHeadcount: React.FC = () => {
  const [organizationData, setOrganizationData] = useState<OrganizationNode[]>([]);
  const [fteData, setFTEData] = useState<FTEData[]>([]);
  const [companies, setCompanies] = useState<string[]>([]);
  const [departments, setDepartments] = useState<string[]>([]);
  const [divisions, setDivisions] = useState<string[]>([]);
  const [sections, setSections] = useState<string[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);  // 초기 선택 없음
  const [selectedDepartment, setSelectedDepartment] = useState<string | null>(null);
  const [selectedDivision, setSelectedDivision] = useState<string | null>(null);
  const [selectedSection, setSelectedSection] = useState<string | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  // 데이터베이스에서 데이터 로드
  useEffect(() => {
    const loadDatabase = async () => {
      try {
        setLoading(true);

        // sql.js 초기화
        const SQL = await initSqlJs({
          locateFile: (file: string) => `https://sql.js.org/dist/${file}`
        });

        // 데이터베이스 파일 로드
        const response = await fetch('/hwaseung_RnD.db');
        const buffer = await response.arrayBuffer();
        const db = new SQL.Database(new Uint8Array(buffer));

        // 조직 데이터 조회
        const orgResult = db.exec(`
          SELECT 회사, 본부, 담당_사업단_센터, 실, 팀
          FROM organization
          WHERE 팀 IS NOT NULL AND 팀 != ''
        `);

        if (orgResult.length > 0) {
          const values = orgResult[0].values;

          const orgData: OrganizationNode[] = values.map((row: any[]) => ({
            회사: row[0],
            본부: row[1],
            담당_사업단_센터: row[2],
            실: row[3],
            팀: row[4]
          }));

          setOrganizationData(orgData);

          // 회사 목록 추출
          const uniqueCompanies = Array.from(
            new Set(orgData.map(org => org.회사).filter(Boolean))
          ) as string[];
          setCompanies(uniqueCompanies);
        }

        // FTE 데이터 조회
        const fteResult = db.exec(`
          SELECT
            팀명, 회사,
            FTE_전체, FTE_책임, FTE_선임, FTE_사원,
            인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
            FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원
          FROM fte
        `);

        if (fteResult.length > 0) {
          const fteValues = fteResult[0].values;

          const fteDataArray: FTEData[] = fteValues.map((row: any[]) => ({
            팀명: row[0],
            회사: row[1],
            FTE_전체: row[2] || 0,
            FTE_책임: row[3] || 0,
            FTE_선임: row[4] || 0,
            FTE_사원: row[5] || 0,
            인원수_전체: row[6] || 0,
            인원수_책임: row[7] || 0,
            인원수_선임: row[8] || 0,
            인원수_사원: row[9] || 0,
            FTE_per_인원_전체: row[10] || 0,
            FTE_per_인원_책임: row[11] || 0,
            FTE_per_인원_선임: row[12] || 0,
            FTE_per_인원_사원: row[13] || 0
          }));

          setFTEData(fteDataArray);
        }

        // 데이터베이스 연결 종료
        db.close();
        setLoading(false);
      } catch (error) {
        console.error('Failed to load database:', error);
        setLoading(false);
      }
    };

    loadDatabase();
  }, []);

  // 회사 선택시 본부 목록 업데이트
  useEffect(() => {
    if (selectedCompany) {
      const uniqueDepartments = Array.from(
        new Set(
          organizationData
            .filter(org => org.회사 === selectedCompany)
            .map(org => org.본부)
            .filter(Boolean)
        )
      );
      setDepartments(uniqueDepartments);
    } else {
      setDepartments([]);
    }
    setSelectedDepartment(null);
    setSelectedDivision(null);
    setSelectedSection(null);
    setSelectedTeam(null);
  }, [selectedCompany, organizationData]);

  // 본부 선택시 담당/사업단/센터 목록 업데이트
  useEffect(() => {
    if (selectedCompany && selectedDepartment) {
      const uniqueDivisions = Array.from(
        new Set(
          organizationData
            .filter(org => org.회사 === selectedCompany && org.본부 === selectedDepartment)
            .map(org => org.담당_사업단_센터)
            .filter(Boolean)
        )
      );
      setDivisions(uniqueDivisions);
    } else {
      setDivisions([]);
    }
    setSelectedDivision(null);
    setSelectedSection(null);
    setSelectedTeam(null);
  }, [selectedDepartment, selectedCompany, organizationData]);

  // 담당/사업단/센터 선택시 실 목록 업데이트
  useEffect(() => {
    if (selectedCompany && selectedDepartment && selectedDivision) {
      const uniqueSections = Array.from(
        new Set(
          organizationData
            .filter(
              org =>
                org.회사 === selectedCompany &&
                org.본부 === selectedDepartment &&
                org.담당_사업단_센터 === selectedDivision
            )
            .map(org => org.실)
            .filter(Boolean)
        )
      );
      setSections(uniqueSections);
    } else {
      setSections([]);
    }
    setSelectedSection(null);
    setSelectedTeam(null);
  }, [selectedDivision, selectedDepartment, selectedCompany, organizationData]);

  // 실 선택시 팀 목록 업데이트
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
                org.실 === selectedSection
            )
            .map(org => org.팀)
            .filter(Boolean)
        )
      );
      setTeams(uniqueTeams);
    } else {
      setTeams([]);
    }
    setSelectedTeam(null);
  }, [selectedSection, selectedDivision, selectedDepartment, selectedCompany, organizationData]);

  // Breadcrumb 네비게이션 핸들러
  const handleBreadcrumbNavigation = (level: string) => {
    switch (level) {
      case 'home':
        setSelectedCompany(null);
        setSelectedDepartment(null);
        setSelectedDivision(null);
        setSelectedSection(null);
        setSelectedTeam(null);
        break;
      case 'company':
        setSelectedDepartment(null);
        setSelectedDivision(null);
        setSelectedSection(null);
        setSelectedTeam(null);
        break;
      case 'department':
        setSelectedDivision(null);
        setSelectedSection(null);
        setSelectedTeam(null);
        break;
      case 'division':
        setSelectedSection(null);
        setSelectedTeam(null);
        break;
      case 'section':
        setSelectedTeam(null);
        break;
      default:
        break;
    }
  };

  // 드릴다운 핸들러
  const handleDrillDown = (orgName: string) => {
    // 현재 선택 레벨에 따라 다음 레벨로 이동
    if (!selectedCompany) {
      // 회사 선택
      setSelectedCompany(orgName);
    } else if (!selectedDepartment) {
      // 본부 선택
      setSelectedDepartment(orgName);
    } else if (!selectedDivision) {
      // 담당/사업단/센터 선택
      setSelectedDivision(orgName);
    } else if (!selectedSection) {
      // 실 선택 - 실이 없는 경우도 있으므로 확인 필요
      const hasSections = organizationData.some(
        org => org.회사 === selectedCompany &&
               org.본부 === selectedDepartment &&
               org.담당_사업단_센터 === selectedDivision &&
               org.실 && org.실 !== ''
      );
      if (hasSections) {
        setSelectedSection(orgName);
      } else {
        // 실이 없으면 바로 팀으로
        setSelectedTeam(orgName);
      }
    } else if (!selectedTeam) {
      // 팀 선택
      setSelectedTeam(orgName);
    }
  };

  // Breadcrumb items 생성
  const breadcrumbItems: BreadcrumbItem[] = [];
  if (selectedCompany) breadcrumbItems.push({ label: selectedCompany, value: 'company' });
  if (selectedDepartment) breadcrumbItems.push({ label: selectedDepartment, value: 'department' });
  if (selectedDivision) breadcrumbItems.push({ label: selectedDivision, value: 'division' });
  if (selectedSection) breadcrumbItems.push({ label: selectedSection, value: 'section' });
  if (selectedTeam) breadcrumbItems.push({ label: selectedTeam, value: 'team' });

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-foreground">인력 수준 검토</h1>
            <p className="text-muted-foreground">근무기록 기반 FTE 분석</p>
          </div>
        </div>
        <div className="flex justify-center items-center h-64">
          <div className="text-gray-500">데이터베이스 로딩 중...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 헤더 */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">인력 수준 검토</h1>
          <p className="text-muted-foreground">
            근무기록 기반 FTE 분석 {!selectedCompany && '- 전체 회사'}
          </p>
        </div>
      </div>

      {/* Miller Column */}
      <Card className="mb-4 overflow-hidden bg-white">
        <div
          className="flex overflow-x-auto bg-white"
          style={{ minWidth: '1000px' }} // 최소 너비 설정으로 5개 컬럼 (200px * 5)
        >
          {/* 회사 Column */}
          <ColumnPane
            title="회사"
            items={companies}
            selectedValue={selectedCompany}
            onSelect={setSelectedCompany}
            width="200px"
          />

          {/* 본부 Column */}
          {selectedCompany && (
            <ColumnPane
              title="본부"
              items={departments}
              selectedValue={selectedDepartment}
              onSelect={setSelectedDepartment}
              width="200px"
            />
          )}

          {/* 담당/사업단/센터 Column */}
          {selectedDepartment && (
            <ColumnPane
              title="담당/사업단/센터"
              items={divisions}
              selectedValue={selectedDivision}
              onSelect={setSelectedDivision}
              width="240px" // 제목이 길어서 약간 넓게
            />
          )}

          {/* 실 Column */}
          {selectedDivision && sections.length > 0 && (
            <ColumnPane
              title="실"
              items={sections}
              selectedValue={selectedSection}
              onSelect={setSelectedSection}
              width="200px"
            />
          )}

          {/* 팀 Column */}
          {((selectedSection && teams.length > 0) ||
            (selectedDivision && !sections.length && teams.length > 0)) && (
            <ColumnPane
              title="팀"
              items={teams}
              selectedValue={selectedTeam}
              onSelect={setSelectedTeam}
              width="200px"
            />
          )}

          {/* 빈 공간 채우기 - 선택되지 않은 컬럼 영역을 미리 확보 */}
          {!selectedCompany && (
            <>
              <div className="w-[200px] flex-shrink-0" />
              <div className="w-[240px] flex-shrink-0" />
              <div className="w-[200px] flex-shrink-0" />
              <div className="w-[200px] flex-shrink-0" />
            </>
          )}
          {selectedCompany && !selectedDepartment && (
            <>
              <div className="w-[240px] flex-shrink-0" />
              <div className="w-[200px] flex-shrink-0" />
              <div className="w-[200px] flex-shrink-0" />
            </>
          )}
          {selectedDepartment && !selectedDivision && (
            <>
              <div className="w-[200px] flex-shrink-0" />
              <div className="w-[200px] flex-shrink-0" />
            </>
          )}
          {selectedDivision && sections.length > 0 && !selectedSection && (
            <div className="w-[200px] flex-shrink-0" />
          )}
        </div>
      </Card>

      {/* Breadcrumb */}
      {breadcrumbItems.length > 0 && (
        <Card className="p-4 bg-white mb-6">
          <Breadcrumb items={breadcrumbItems} onNavigate={handleBreadcrumbNavigation} />
        </Card>
      )}

      {/* FTE 통계 */}
      <FTEStatistics
        selectedLevel={{
          company: selectedCompany || undefined,
          department: selectedDepartment || undefined,
          division: selectedDivision || undefined,
          section: selectedSection || undefined,
          team: selectedTeam || undefined
        }}
        organizationData={organizationData}
        fteData={fteData}
        onDrillDown={handleDrillDown}
      />
    </div>
  );
};

export default OrganizationHeadcount;