import React from 'react';
import { Switch } from '../components/ui/switch';
import { Card } from '../components/ui/card';
import { Label } from '../components/ui/label';
import { Bell, Mail, Smartphone } from 'lucide-react';

interface NotificationPreference {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
}

const preferences: NotificationPreference[] = [
  {
    id: 'trading-signals',
    label: 'Trading Signals',
    description: 'Get notified about high-confidence trading opportunities',
    icon: <Bell className="h-4 w-4" />
  },
  {
    id: 'email-notifications',
    label: 'Email Notifications',
    description: 'Receive detailed trading signals via email',
    icon: <Mail className="h-4 w-4" />
  },
  {
    id: 'mobile-notifications',
    label: 'Mobile Notifications',
    description: 'Push notifications for urgent trading signals',
    icon: <Smartphone className="h-4 w-4" />
  }
];

export const NotificationPreferences: React.FC = () => {
  const [enabled, setEnabled] = React.useState<Record<string, boolean>>({
    'trading-signals': true,
    'email-notifications': true,
    'mobile-notifications': false
  });

  const handleToggle = (id: string) => {
    setEnabled(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="space-y-1">
        <h3 className="text-lg font-medium">Notification Preferences</h3>
        <p className="text-sm text-gray-500">
          Customize how you receive trading signals and market updates
        </p>
      </div>

      <div className="space-y-4">
        {preferences.map(({ id, label, description, icon }) => (
          <div key={id} className="flex items-center justify-between space-x-4">
            <div className="flex items-center space-x-4">
              <div className="p-2 rounded-full bg-primary/10">
                {icon}
              </div>
              <div className="space-y-1">
                <Label htmlFor={id} className="text-sm font-medium">
                  {label}
                </Label>
                <p className="text-sm text-gray-500">{description}</p>
              </div>
            </div>
            <Switch
              id={id}
              checked={enabled[id]}
              onCheckedChange={() => handleToggle(id)}
            />
          </div>
        ))}
      </div>
    </Card>
  );
};
