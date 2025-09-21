import React from 'react';
import { cn } from '../../lib/utils';

interface MagicCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  gradientColor?: string;
}

export function MagicCard({
  children,
  className,
  gradientColor = 'from-gray-50 to-white',
  ...props
}: MagicCardProps) {
  return (
    <div
      className={cn(
        'relative rounded-xl border border-gray-200 bg-gradient-to-br shadow-sm transition-all hover:shadow-md',
        gradientColor,
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}