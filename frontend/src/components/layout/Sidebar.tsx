import React from 'react';
import { NavLink } from 'react-router-dom';
import { Moon, Sun } from 'lucide-react';
import { Button } from '../ui/button';

interface SidebarProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

interface NavItem {
  name: string;
  href?: string;
  isGroup?: boolean;
  children?: { name: string; href: string }[];
}

const navigation: NavItem[] = [
  { name: 'Data 업로드', href: '/data' },
  { name: '전사 모델링', href: '/company-wide-modeling' },
  {
    name: '전사 적정인력 산정/예측',
    isGroup: true,
    children: [
      { name: 'R&A', href: '/dashboard/rna' },
      { name: '통합기술본부', href: '/dashboard/tonggibon' }
    ]
  },
  { name: '조직, 직급별 적정인력 요약', href: '/position-analysis' },
  { name: '조직, 직급별 적정인력 산정/예측', href: '/organization-simulation' },
  { name: '인력 수준 검토', href: '/organization-headcount' },
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
          <div key={item.name}>
            {item.isGroup ? (
              <>
                <div className="px-4 py-2 text-sm font-medium text-muted-foreground">
                  {item.name}
                </div>
                {item.children?.map((child) => (
                  <NavLink
                    key={child.name}
                    to={child.href}
                    className={({ isActive }) =>
                      `flex items-center pl-8 pr-4 py-2 text-sm font-medium rounded-md transition-colors ${
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      }`
                    }
                  >
                    {child.name}
                  </NavLink>
                ))}
              </>
            ) : item.href ? (
              <NavLink
                to={item.href}
                className={({ isActive }) =>
                  `flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  }`
                }
              >
                {item.name}
              </NavLink>
            ) : null}
          </div>
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