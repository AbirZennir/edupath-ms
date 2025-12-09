import { GraduationCap, LayoutDashboard, BookOpen, AlertTriangle, Lightbulb, BarChart3, Settings, LogOut } from 'lucide-react';

export default function Sidebar({ currentPage, onNavigate, onLogout }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'classes', label: 'Classes & Modules', icon: BookOpen },
    { id: 'at-risk', label: 'Étudiants à risque', icon: AlertTriangle },
    { id: 'recommendations', label: 'Recommandations', icon: Lightbulb },
    { id: 'analytics', label: 'Analytics / Rapports', icon: BarChart3 },
    { id: 'settings', label: 'Paramètres', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-white border-r border-[#E2E8F0] min-h-screen flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-[#E2E8F0]">
        <div className="flex items-center gap-3">
          <div className="bg-[#2563EB] p-2 rounded-lg">
            <GraduationCap className="w-6 h-6 text-white" />
          </div>
          <span className="text-[#1E293B]">EduPath-MS</span>
        </div>
      </div>

      {/* Menu */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            return (
              <li key={item.id}>
                <button
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition ${
                    isActive
                      ? 'bg-[#EFF6FF] text-[#2563EB]'
                      : 'text-[#64748B] hover:bg-[#F8FAFC]'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Avatar enseignant et déconnexion */}
      <div className="p-4 border-t border-[#E2E8F0]">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-[#2563EB] flex items-center justify-center text-white">
            MP
          </div>
          <div className="flex-1">
            <p className="text-[#1E293B]">Marie Petit</p>
            <p className="text-[#94A3B8]">Enseignante</p>
          </div>
        </div>
        <button
          onClick={onLogout}
          className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-[#EF4444] hover:bg-[#FEE2E2] transition"
        >
          <LogOut className="w-5 h-5" />
          <span>Déconnexion</span>
        </button>
      </div>
    </aside>
  );
}


