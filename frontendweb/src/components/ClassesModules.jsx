import { Search, Filter, Users, TrendingUp } from 'lucide-react';
import Sidebar from './Sidebar';
import { useState, useEffect } from 'react';
import { api } from '../api/client';

export default function ClassesModules({ onNavigate, onSelectClass, onLogout, user, token }) {
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      fetchModules();
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [searchTerm, token]);

  async function fetchModules() {
    try {
      setLoading(true);
      const params = searchTerm ? { q: searchTerm } : {};
      const data = await api.listCourses(params, token);
      setModules(data);
    } catch (err) {
      console.error("Failed to fetch modules", err);
      setError("Impossible de charger les modules.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex">
      <Sidebar currentPage="classes" onNavigate={onNavigate} onLogout={onLogout} user={user} />

      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-6">Mes modules</h1>

          {/* Barre de recherche et filtres */}
          <div className="flex gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-[#94A3B8]" />
              <input
                type="text"
                placeholder="Rechercher un module..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-12 pr-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white"
              />
            </div>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center py-10 text-[#64748B]">Chargement des modules...</div>
        ) : error ? (
          <div className="text-center py-10 text-red-500">{error}</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {modules.map((module) => (
              <div
                key={module.id}
                className="bg-white rounded-2xl p-6 shadow-sm hover:shadow-md transition cursor-pointer"
                onClick={() => onSelectClass(module.id)}
              >
                {/* Titre et semestre */}
                <div className="mb-4">
                  <h3 className="text-[#1E293B] mb-2 font-semibold">{module.title || module.codeModule}</h3>
                  <p className="text-[#94A3B8]">{module.codePresentation}</p>
                </div>

                {/* Statistiques */}
                <div className="space-y-3 mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[#64748B]">
                      <Users className="w-4 h-4" />
                      <span>Étudiants</span>
                    </div>
                    <span className="text-[#1E293B]">{module.studentsCount}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[#64748B]">
                      <TrendingUp className="w-4 h-4" />
                      <span>Taux de réussite</span>
                    </div>
                    <span className="text-[#1E293B]">{module.successRate}%</span>
                  </div>
                </div>

                {/* Barre de progression */}
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-[#64748B]">Complétion du module</span>
                    <span className="text-[#1E293B]">{module.completion}%</span>
                  </div>
                  <div className="w-full bg-[#E2E8F0] rounded-full h-2">
                    <div
                      className="bg-[#2563EB] h-2 rounded-full transition-all"
                      style={{ width: `${module.completion}%` }}
                    ></div>
                  </div>
                </div>

                {/* Bouton */}
                <button className="w-full py-2 bg-[#EFF6FF] text-[#2563EB] rounded-lg hover:bg-[#DBEAFE] transition">
                  Voir détails
                </button>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
