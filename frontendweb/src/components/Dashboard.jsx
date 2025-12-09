import { RefreshCw, Download, TrendingUp, Users, Activity, BookOpen } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const evolutionData = [
  { semaine: 'S1', reussite: 75, engagement: 68 },
  { semaine: 'S2', reussite: 78, engagement: 72 },
  { semaine: 'S3', reussite: 76, engagement: 70 },
  { semaine: 'S4', reussite: 80, engagement: 75 },
  { semaine: 'S5', reussite: 82, engagement: 78 },
  { semaine: 'S6', reussite: 85, engagement: 82 },
];

const profileData = [
  { name: 'Assidu', value: 35, color: '#22C55E' },
  { name: 'Procrastinateur', value: 25, color: '#F97316' },
  { name: 'En difficulté', value: 20, color: '#EF4444' },
  { name: 'Très performant', value: 20, color: '#2563EB' },
];

const atRiskStudents = [
  { id: 1, nom: 'Dubois Alexandre', classe: 'L3 Info', probabilite: 78, derniereConnexion: '2 jours', statut: 'critique' },
  { id: 2, nom: 'Martin Sophie', classe: 'L3 Info', probabilite: 65, derniereConnexion: '1 jour', statut: 'attention' },
  { id: 3, nom: 'Bernard Lucas', classe: 'L3 Info', probabilite: 72, derniereConnexion: '3 jours', statut: 'critique' },
  { id: 4, nom: 'Petit Emma', classe: 'L3 Info', probabilite: 58, derniereConnexion: '5 heures', statut: 'attention' },
  { id: 5, nom: 'Roux Thomas', classe: 'L3 Info', probabilite: 81, derniereConnexion: '4 jours', statut: 'critique' },
  { id: 6, nom: 'Moreau Léa', classe: 'L3 Info', probabilite: 62, derniereConnexion: '1 jour', statut: 'attention' },
  { id: 7, nom: 'Simon Hugo', classe: 'L3 Info', probabilite: 69, derniereConnexion: '2 jours', statut: 'attention' },
  { id: 8, nom: 'Laurent Chloé', classe: 'L3 Info', probabilite: 75, derniereConnexion: '3 jours', statut: 'critique' },
];

export default function Dashboard({ onNavigate, onLogout }) {
  return (
    <div className="flex">
      <Sidebar currentPage="dashboard" onNavigate={onNavigate} onLogout={onLogout} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h1 className="text-[#1E293B] mb-2">Vue d&apos;ensemble de la classe</h1>
              <div className="flex items-center gap-3">
                <select className="px-4 py-2 rounded-lg border border-[#E2E8F0] bg-white text-[#334155] focus:outline-none focus:ring-2 focus:ring-[#2563EB]">
                  <option>L3 Informatique – Algorithmes</option>
                  <option>L2 Physique – Mécanique</option>
                  <option>M1 Data Science – Machine Learning</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3">
              <button className="flex items-center gap-2 px-4 py-2 bg-white border border-[#E2E8F0] rounded-lg text-[#334155] hover:bg-[#F8FAFC] transition">
                <RefreshCw className="w-4 h-4" />
                Actualiser
              </button>
              <button className="flex items-center gap-2 px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition">
                <Download className="w-4 h-4" />
                Exporter PDF
              </button>
            </div>
          </div>
        </div>

        {/* Cartes de statistiques */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#FEE2E2] p-3 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-[#EF4444]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Étudiants à risque</p>
            <p className="text-[#1E293B]">15</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#DBEAFE] p-3 rounded-lg">
                <TrendingUp className="w-6 h-6 text-[#2563EB]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Taux de réussite global</p>
            <p className="text-[#1E293B]">85%</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#DCFCE7] p-3 rounded-lg">
                <Activity className="w-6 h-6 text-[#22C55E]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Engagement moyen</p>
            <div className="mt-2">
              <div className="w-full bg-[#E2E8F0] rounded-full h-2">
                <div className="bg-[#22C55E] h-2 rounded-full" style={{ width: '82%' }}></div>
              </div>
              <p className="text-[#1E293B] mt-1">82%</p>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#FED7AA] p-3 rounded-lg">
                <BookOpen className="w-6 h-6 text-[#F97316]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Ressources consultées</p>
            <p className="text-[#1E293B]">127</p>
          </div>
        </div>

        {/* Graphiques */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Graphique de courbes */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Évolution dans le temps</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={evolutionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="semaine" stroke="#64748B" />
                <YAxis stroke="#64748B" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="reussite" stroke="#2563EB" strokeWidth={2} name="Taux de réussite" />
                <Line type="monotone" dataKey="engagement" stroke="#22C55E" strokeWidth={2} name="Engagement" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Graphique en camembert */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Répartition des profils étudiants</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={profileData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {profileData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Tableau étudiants à risque */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <h3 className="text-[#1E293B] mb-4">Top 10 étudiants à risque</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[#E2E8F0]">
                  <th className="text-left py-3 px-4 text-[#64748B]">Nom</th>
                  <th className="text-left py-3 px-4 text-[#64748B]">Classe</th>
                  <th className="text-left py-3 px-4 text-[#64748B]">Probabilité d&apos;échec</th>
                  <th className="text-left py-3 px-4 text-[#64748B]">Dernière connexion</th>
                  <th className="text-left py-3 px-4 text-[#64748B]">Statut</th>
                </tr>
              </thead>
              <tbody>
                {atRiskStudents.map((student) => (
                  <tr key={student.id} className="border-b border-[#F1F5F9] hover:bg-[#F8FAFC]">
                    <td className="py-3 px-4 text-[#1E293B]">{student.nom}</td>
                    <td className="py-3 px-4 text-[#64748B]">{student.classe}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-[#E2E8F0] rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              student.probabilite > 70 ? 'bg-[#EF4444]' : 'bg-[#F97316]'
                            }`}
                            style={{ width: `${student.probabilite}%` }}
                          ></div>
                        </div>
                        <span className="text-[#1E293B]">{student.probabilite}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-[#64748B]">{student.derniereConnexion}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-3 py-1 rounded-full ${
                          student.statut === 'critique'
                            ? 'bg-[#FEE2E2] text-[#EF4444]'
                            : 'bg-[#FED7AA] text-[#F97316]'
                        }`}
                      >
                        {student.statut === 'critique' ? 'Critique' : 'Attention'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

function AlertTriangle({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  );
}


