import React, { useState } from 'react';
import { Card, CardContent } from '../components/ui/card';
import { CompanyWideUpload } from '../components/upload/CompanyWideUpload';
import { TeamUpload } from '../components/upload/TeamUpload';
import { FTEUpload } from '../components/upload/FTEUpload';
import { OrganizationChartUpload } from '../components/upload/OrganizationChartUpload';
import { Building2, Factory, Users, Clock, Network } from 'lucide-react';

type TabType = 'rna' | 'tonggibon' | 'team' | 'fte' | 'organization';

export const DataUpload: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('rna');

  const tabs = [
    {
      id: 'rna' as TabType,
      label: '전사-R*A',
      icon: Building2,
    },
    {
      id: 'tonggibon' as TabType,
      label: '전사-통기본',
      icon: Factory,
    },
    {
      id: 'team' as TabType,
      label: '조직인력',
      icon: Users,
    },
    {
      id: 'fte' as TabType,
      label: 'FTE분석',
      icon: Clock,
    },
    {
      id: 'organization' as TabType,
      label: '조직도',
      icon: Network,
    },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* 페이지 헤더 */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">데이터 업로드</h1>
        <p className="text-muted-foreground mt-2">
          적정인력 산정을 위한 데이터를 업로드하세요
        </p>
      </div>

      {/* 탭 네비게이션 */}
      <Card>
        <CardContent className="p-0">
          <div className="flex border-b overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-6 py-4 font-medium transition-colors border-b-2 whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-primary text-primary bg-primary/5'
                      : 'border-transparent text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* 탭 컨텐츠 */}
      <div className="mt-6">
        {activeTab === 'rna' && (
          <CompanyWideUpload
            organization="R*A"
            title="전사 인력 산정 데이터 (R*A)"
            description="R*A 조직의 전사 수준 적정인력 예측을 위한 데이터를 업로드하세요"
          />
        )}

        {activeTab === 'tonggibon' && (
          <CompanyWideUpload
            organization="tonggibon"
            title="전사 인력 산정 데이터 (통합기술본부)"
            description="통합기술본부 조직의 전사 수준 적정인력 예측을 위한 데이터를 업로드하세요"
          />
        )}

        {activeTab === 'team' && (
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-2">조직인력 산정 데이터</h2>
              <p className="text-muted-foreground mb-4">
                팀별, 월별 적정인력 예측을 위한 Feature 데이터를 업로드하세요
              </p>
              <TeamUpload />
            </CardContent>
          </Card>
        )}

        {activeTab === 'fte' && (
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-2">FTE 분석 데이터</h2>
              <p className="text-muted-foreground mb-4">
                팀별 평균 FTE 및 업무 강도 분석 데이터를 업로드하세요
              </p>
              <FTEUpload />
            </CardContent>
          </Card>
        )}

        {activeTab === 'organization' && (
          <Card>
            <CardContent className="p-6">
              <h2 className="text-2xl font-bold mb-2">조직도 데이터</h2>
              <p className="text-muted-foreground mb-4">
                조직 구조 정보를 업데이트하세요 (HQ, 본부, 담당/사업단/센터, 실, 팀)
              </p>
              <OrganizationChartUpload />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};
