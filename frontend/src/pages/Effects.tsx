import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';

export const Effects: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">기대효과</h1>
        <p className="text-muted-foreground">시나리오별 기대효과 분석 및 리포트</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>경제적 효과</CardTitle>
            <CardDescription>임금인상의 경제적 파급효과</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span>소비 증가율</span>
                <span className="font-semibold">+2.1%</span>
              </div>
              <div className="flex justify-between">
                <span>물가 상승율</span>
                <span className="font-semibold">+0.8%</span>
              </div>
              <div className="flex justify-between">
                <span>고용 증가율</span>
                <span className="font-semibold">+1.3%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>사회적 효과</CardTitle>
            <CardDescription>임금인상의 사회적 파급효과</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span>소득 격차 개선</span>
                <span className="font-semibold text-green-600">개선</span>
              </div>
              <div className="flex justify-between">
                <span>생활 만족도</span>
                <span className="font-semibold text-green-600">향상</span>
              </div>
              <div className="flex justify-between">
                <span>사회 안정성</span>
                <span className="font-semibold text-green-600">증대</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>종합 리포트</CardTitle>
          <CardDescription>전체 분석 결과 및 권고사항</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 bg-background border rounded-md flex items-center justify-center">
            <p className="text-muted-foreground">리포트 생성 영역</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};