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

          console.log('✅ Organization data loaded:', orgData.length, 'rows');
          setOrganizationData(orgData);

          // 회사 목록 추출
          const uniqueCompanies = Array.from(
            new Set(orgData.map(org => org.회사).filter(Boolean))
          ) as string[];
          console.log('✅ Companies:', uniqueCompanies);
          setCompanies(uniqueCompanies);
        } else {
          console.warn('⚠️ No organization data found');
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

          console.log('✅ FTE data loaded:', fteDataArray.length, 'rows');
          console.log('Sample FTE data:', fteDataArray[0]);
          setFTEData(fteDataArray);
        } else {
          console.warn('⚠️ No FTE data found');
        }

        // 데이터베이스 연결 종료
        db.close();
        setLoading(false);
      } catch (error) {
        console.error('❌ Failed to load database:', error);
        console.error('Error details:', error instanceof Error ? error.message : String(error));
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

      {/* Miller Column Navigation */}
      <div className="grid grid-cols-5 gap-4">
        {/* 회사 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">회사</h3>
          <div className="space-y-2">
            {companies.map((company) => (
              <button
                key={company}
                onClick={() => setSelectedCompany(company)}
                className={`w-full px-3 py-2 text-left rounded text-sm ${
                  selectedCompany === company
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'hover:bg-gray-50 border border-gray-200'
                }`}
              >
                {company}
              </button>
            ))}
          </div>
        </Card>

        {/* 본부 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">본부</h3>
          <div className="space-y-2">
            {selectedCompany ? (
              departments.length > 0 ? (
                departments.map((dept) => (
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
                ))
              ) : (
                <div className="text-sm text-gray-400 text-center py-2">데이터 없음</div>
              )
            ) : (
              <div className="text-sm text-gray-400 text-center py-2">회사를 선택하세요</div>
            )}
          </div>
        </Card>

        {/* 담당/사업단/센터 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">담당/사업단/센터</h3>
          <div className="space-y-2">
            {selectedDepartment ? (
              divisions.length > 0 ? (
                divisions.map((div) => (
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
                ))
              ) : (
                <div className="text-sm text-gray-400 text-center py-2">데이터 없음</div>
              )
            ) : (
              <div className="text-sm text-gray-400 text-center py-2">본부를 선택하세요</div>
            )}
          </div>
        </Card>

        {/* 실 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">실</h3>
          <div className="space-y-2">
            {selectedDivision ? (
              sections.length > 0 ? (
                sections.map((sec) => (
                  <button
                    key={sec}
                    onClick={() => setSelectedSection(sec)}
                    className={`w-full px-3 py-2 text-left rounded text-sm ${
                      selectedSection === sec
                        ? 'bg-blue-100 text-blue-800 border border-blue-300'
                        : 'hover:bg-gray-50 border border-gray-200'
                    }`}
                  >
                    {sec}
                  </button>
                ))
              ) : (
                <div className="text-sm text-gray-400 text-center py-2">데이터 없음</div>
              )
            ) : (
              <div className="text-sm text-gray-400 text-center py-2">담당을 선택하세요</div>
            )}
          </div>
        </Card>

        {/* 팀 */}
        <Card className="p-4 bg-white">
          <h3 className="font-medium text-gray-700 mb-3 text-center">팀</h3>
          <div className="space-y-2">
            {selectedSection || (selectedDivision && sections.length === 0) ? (
              teams.length > 0 ? (
                teams.map((team) => (
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
                ))
              ) : (
                <div className="text-sm text-gray-400 text-center py-2">데이터 없음</div>
              )
            ) : (
              <div className="text-sm text-gray-400 text-center py-2">실을 선택하세요</div>
            )}
          </div>
        </Card>
      </div>

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