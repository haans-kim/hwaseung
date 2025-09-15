import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';

interface DepartmentData {
  name: string;
  senior: { value: number; type: 'up' | 'down' | 'same' };
  deputy: { value: number; type: 'up' | 'down' | 'same' };
  manager: { value: number; type: 'up' | 'down' | 'same' };
}

const PositionAnalysis: React.FC = () => {
  const [departmentData] = useState<DepartmentData[]>([
    {
      name: '구부',
      senior: { value: 1, type: 'same' },
      deputy: { value: 2, type: 'same' },
      manager: { value: 2, type: 'down' }
    },
    {
      name: 'SL제로팀',
      senior: { value: 2, type: 'same' },
      deputy: { value: 1, type: 'down' },
      manager: { value: 1, type: 'down' }
    },
    {
      name: 'FL제로팀',
      senior: { value: 6, type: 'up' },
      deputy: { value: 3, type: 'same' },
      manager: { value: 8, type: 'up' }
    },
    {
      name: '인더스트리얼...',
      senior: { value: 3, type: 'same' },
      deputy: { value: 6, type: 'up' },
      manager: { value: 7, type: 'up' }
    },
    {
      name: 'SPECIALTY...',
      senior: { value: 2, type: 'same' },
      deputy: { value: 4, type: 'same' },
      manager: { value: 5, type: 'same' }
    },
    {
      name: '선행개발팀',
      senior: { value: 1, type: 'down' },
      deputy: { value: 3, type: 'same' },
      manager: { value: 4, type: 'same' }
    },
    {
      name: '성능개발팀',
      senior: { value: 2, type: 'same' },
      deputy: { value: 5, type: 'up' },
      manager: { value: 3, type: 'same' }
    },
    {
      name: 'SL생산기술팀',
      senior: { value: 3, type: 'same' },
      deputy: { value: 4, type: 'same' },
      manager: { value: 6, type: 'up' }
    },
    {
      name: 'FL생산기술팀',
      senior: { value: 3, type: 'same' },
      deputy: { value: 4, type: 'same' },
      manager: { value: 6, type: 'up' }
    }
  ]);

  const getCellStyle = (type: 'up' | 'down' | 'same') => {
    switch (type) {
      case 'up':
        return 'bg-red-200 text-red-800';
      case 'down':
        return 'bg-blue-200 text-blue-800';
      case 'same':
        return 'bg-green-200 text-green-800';
    }
  };

  const getIcon = (type: 'up' | 'down' | 'same') => {
    switch (type) {
      case 'up':
        return '▲';
      case 'down':
        return '▼';
      case 'same':
        return '●';
    }
  };

  const getTotalCurrent = () => 600;
  const getTotalHireNeeded = () => 5;
  const getTotalPromoteNeeded = () => 10;
  const getTotalTransferNeeded = () => 18;

  return (
    <div className="min-h-screen bg-white p-6">
      {/* 헤더 */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-foreground mb-2">직급별 분석</h1>
        <p className="text-muted-foreground">직급별 적정인원 도출</p>
      </div>


      {/* 통계 카드들 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="p-4 text-center border shadow-sm bg-white">
          <div className="text-2xl font-bold text-black mb-2">600</div>
          <div className="text-sm text-gray-600">총 분석 인원</div>
        </Card>
        <Card className="p-4 text-center border shadow-sm bg-white">
          <div className="text-2xl font-bold text-black mb-2">5명</div>
          <div className="text-sm text-gray-600">선임 충원 필요</div>
        </Card>
        <Card className="p-4 text-center border shadow-sm bg-white">
          <div className="text-2xl font-bold text-black mb-2">10명</div>
          <div className="text-sm text-gray-600">책임 충원 필요</div>
        </Card>
        <Card className="p-4 text-center border shadow-sm bg-white">
          <div className="text-2xl font-bold text-black mb-2">18명</div>
          <div className="text-sm text-gray-600">사원 충원 필요</div>
        </Card>
      </div>

      {/* 직급별 적정인원(예측) 테이블 카드 */}
      <Card className="border shadow-lg bg-white">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-bold">직급별 적정인원(예측)</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            {/* 헤더 */}
            <div className="grid grid-cols-9 border-b">
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center flex items-center justify-center">구분</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">SL재료팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">FL재료팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">인더스트리얼재료팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">SPECIALTY...</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">선행개발팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">성능개발팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">SL생산기술팀</div>
              <div className="py-3 px-4 font-medium text-muted-foreground text-sm text-center border-l">FL생산기술팀</div>
            </div>

            {/* 선임 행 */}
            <div className="grid grid-cols-9 border-b">
              <div className="py-4 px-4 font-medium text-muted-foreground text-sm text-center flex items-center justify-center">선임</div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  2 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  6 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  2 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-blue-200 bg-blue-50 text-blue-800 text-sm font-medium">
                  1 ▼
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  2 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
            </div>

            {/* 책임 행 */}
            <div className="grid grid-cols-9 border-b">
              <div className="py-4 px-4 font-medium text-muted-foreground text-sm text-center flex items-center justify-center">책임</div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-blue-200 bg-blue-50 text-blue-800 text-sm font-medium">
                  1 ▼
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  6 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  4 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  5 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  4 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  4 ●
                </div>
              </div>
            </div>

            {/* 사원 행 */}
            <div className="grid grid-cols-9">
              <div className="py-4 px-4 font-medium text-muted-foreground text-sm text-center flex items-center justify-center">사원</div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-blue-200 bg-blue-50 text-blue-800 text-sm font-medium">
                  1 ▼
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  8 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  7 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  5 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  4 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-green-200 bg-green-50 text-green-800 text-sm font-medium">
                  3 ●
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  6 ▲
                </div>
              </div>
              <div className="py-4 px-4 text-center border-l">
                <div className="inline-flex items-center justify-center w-24 px-3 py-2 rounded-lg border-2 border-red-200 bg-red-50 text-red-800 text-sm font-medium">
                  6 ▲
                </div>
              </div>
            </div>
          </div>

          {/* 범례 */}
          <div className="mt-6 pt-4 border-t flex items-center gap-6 text-sm text-muted-foreground px-4 pb-4">
            <div className="flex items-center gap-2">
              <span className="text-red-600">▲</span>
              <span>충원필요(부족)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-green-600">●</span>
              <span>적정(증감)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-blue-600">▼</span>
              <span>감원검토(남음)</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PositionAnalysis;