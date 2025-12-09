import { Search, Filter, Users, TrendingUp } from 'lucide-react';
import Sidebar from './Sidebar';

const classes = [
  {
    id: '1',
    title: 'L3 Informatique – Algorithmique',
    students: 45,
    successRate: 85,
    completion: 78,
    semester: 'Semestre 5',
  },
  {
    id: '2',
    title: 'L2 Physique – Mécanique',
    students: 38,
    successRate: 78,
    completion: 65,
    semester: 'Semestre 4',
  },
  {
    id: '3',
    title: 'M1 Data Science – Machine Learning',
    students: 32,
    successRate: 92,
    completion: 88,
    semester: 'Semestre 7',
  },
  {
    id: '4',
    title: 'L3 Informatique – Base de données',
    students: 42,
    successRate: 81,
    completion: 72,
    semester: 'Semestre 5',
  },
  {
    id: '5',
    title: 'L2 Mathématiques – Algèbre linéaire',
    students: 50,
    successRate: 75,
    completion: 68,
    semester: 'Semestre 3',
  },
  {
    id: '6',
    title: 'M2 Intelligence Artificielle – Deep Learning',
    students: 28,
    successRate: 89,
    completion: 82,
    semester: 'Semestre 9',
  },
];

export default function ClassesModules({ onNavigate, onSelectClass, onLogout }) {
  return (
    <div className="flex">
      <Sidebar currentPage="classes" onNavigate={onNavigate} onLogout={onLogout} />

      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-6">Mes classes</h1>

          {/* Barre de recherche et filtres */}
          <div className="flex gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-[#94A3B8]" />
              <input
                type="text"
                placeholder="Rechercher une classe ou un module..."
                className="w-full pl-12 pr-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white"
              />
            </div>
            <button className="flex items-center gap-2 px-6 py-3 bg-white border border-[#E2E8F0] rounded-lg text-[#334155] hover:bg-[#F8FAFC] transition">
              <Filter className="w-5 h-5" />
              Filtres
            </button>
          </div>

          {/* Filtres rapides */}
          <div className="flex gap-3">
            <button className="px-4 py-2 bg-[#EFF6FF] text-[#2563EB] rounded-lg">
              Toutes les classes
            </button>
            <button className="px-4 py-2 bg-white border border-[#E2E8F0] text-[#64748B] rounded-lg hover:bg-[#F8FAFC] transition">
              Licence
            </button>
            <button className="px-4 py-2 bg-white border border-[#E2E8F0] text-[#64748B] rounded-lg hover:bg-[#F8FAFC] transition">
              Master
            </button>
            <button className="px-4 py-2 bg-white border border-[#E2E8F0] text-[#64748B] rounded-lg hover:bg-[#F8FAFC] transition">
              Informatique
            </button>
          </div>
        </div>

        {/* Grille de cartes */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {classes.map((classe) => (
            <div
              key={classe.id}
              className="bg-white rounded-2xl p-6 shadow-sm hover:shadow-md transition cursor-pointer"
              onClick={() => onSelectClass(classe.id)}
            >
              {/* Titre et semestre */}
              <div className="mb-4">
                <h3 className="text-[#1E293B] mb-2">{classe.title}</h3>
                <p className="text-[#94A3B8]">{classe.semester}</p>
              </div>

              {/* Statistiques */}
              <div className="space-y-3 mb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-[#64748B]">
                    <Users className="w-4 h-4" />
                    <span>Étudiants</span>
                  </div>
                  <span className="text-[#1E293B]">{classe.students}</span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-[#64748B]">
                    <TrendingUp className="w-4 h-4" />
                    <span>Taux de réussite</span>
                  </div>
                  <span className="text-[#1E293B]">{classe.successRate}%</span>
                </div>
              </div>

              {/* Barre de progression */}
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[#64748B]">Complétion du module</span>
                  <span className="text-[#1E293B]">{classe.completion}%</span>
                </div>
                <div className="w-full bg-[#E2E8F0] rounded-full h-2">
                  <div
                    className="bg-[#2563EB] h-2 rounded-full transition-all"
                    style={{ width: `${classe.completion}%` }}
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
      </main>
    </div>
  );
}
