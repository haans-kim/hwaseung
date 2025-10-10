import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Upload,
  Settings,
  Activity,
  Users,
  Moon,
  Sun,
  LineChart,
  Building2,
  SlidersHorizontal
} from 'lucide-react';
import { Button } from '../ui/button';

interface SidebarProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const navigation = [
  { name: 'Data 업로드', href: '/data', icon: Upload },
  { name: '모델링', href: '/modeling', icon: Settings },
  { name: '전사 적정인력 산정/예측 (R&A)', href: '/dashboard/rna', icon: Activity },
  { name: '전사 적정인력 산정/예측 (통합기술본부)', href: '/dashboard/tonggibon', icon: Activity },
  { name: '조직, 직급별 적정인력 요약', href: '/position-analysis', icon: Users },
  { name: '조직, 직급별 적정인력 산정/예측', href: '/organization-simulation', icon: SlidersHorizontal },
  { name: '인력 수준 검토', href: '/organization-headcount', icon: Building2 },
];

export const Sidebar: React.FC<SidebarProps> = ({ isDarkMode, toggleDarkMode }) => {
  return (
    <div className="w-64 bg-background border-r border-border h-screen flex flex-col flex-shrink-0">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <h1 className="text-xl font-bold text-foreground">Headcount Optimization</h1>
        <p className="text-sm text-muted-foreground">적정인력 산정 시스템</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              }`
            }
          >
            <item.icon className="mr-3 h-5 w-5" />
            {item.name}
          </NavLink>
        ))}
      </nav>

      {/* ExplainerDashboard Link */}
      <div className="px-4 pb-2">
        <NavLink
          to="/explainer"
          className={({ isActive }) =>
            `flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              isActive
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
            }`
          }
        >
          <LineChart className="mr-3 h-5 w-5" />
          Explainer Dashboard
        </NavLink>
      </div>

      {/* Theme Toggle */}
      <div className="p-4 border-t border-border">
        <Button
          variant="outline"
          size="sm"
          onClick={toggleDarkMode}
          className="w-full"
        >
          {isDarkMode ? (
            <>
              <Sun className="mr-2 h-4 w-4" />
              Light Mode
            </>
          ) : (
            <>
              <Moon className="mr-2 h-4 w-4" />
              Dark Mode
            </>
          )}
        </Button>
      </div>
    </div>
  );
};