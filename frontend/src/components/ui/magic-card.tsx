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
        'relative rounded-lg border shadow-sm transition-all hover:shadow-md',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}