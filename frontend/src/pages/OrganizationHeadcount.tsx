import React from 'react';
import { Card } from '../components/ui/card';
import { Building2 } from 'lucide-react';

const OrganizationHeadcount: React.FC = () => {
  return (
    <div className="min-h-screen bg-white p-6">
      {/* 헤더 */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-foreground mb-2">조직별 적정인원</h1>
        <p className="text-muted-foreground">FTE 분석 기반 적정인원 산출</p>
      </div>

      {/* 임시 콘텐츠 */}
      <Card className="p-8">
        <div className="flex items-center justify-center flex-col space-y-4">
          <Building2 className="h-16 w-16 text-muted-foreground" />
          <p className="text-muted-foreground">준비 중입니다</p>
        </div>
      </Card>
    </div>
  );
};

export default OrganizationHeadcount;