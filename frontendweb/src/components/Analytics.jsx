import { Download, TrendingUp, Activity, Users, Clock } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const moduleSuccessData = [
  { module: 'Algorithmes', taux: 85 },
  { module: 'Base de données', taux: 78 },
  { module: 'Réseaux', taux: 82 },
  { module: 'Web Dev', taux: 91 },
  { module: 'IA', taux: 88 },
  { module: 'Sécurité', taux: 75 },
];

const profileDistributionData = [
  { profil: 'Assidu', count: 35 },
  { profil: 'Procrastinateur', count: 25 },
  { profil: 'En difficulté', count: 20 },
  { profil: 'Très performant', count: 20 },
];

const weeklyActivityData = [
  { jour: 'Lun', h8: 12, h10: 25, h14: 35, h16: 28, h18: 15 },
  { jour: 'Mar', h8: 15, h10: 30, h14: 38, h16: 32, h18: 18 },
  { jour: 'Mer', h8: 18, h10: 35, h14: 42, h16: 30, h18: 12 },
  { jour: 'Jeu', h8: 14, h10: 28, h14: 40, h16: 35, h18: 20 },
  { jour: 'Ven', h8: 10, h10: 22, h14: 30, h16: 25, h18: 10 },
];

const engagementTrendData = [
  { semaine: 'S1', engagement: 68, reussite: 75 },
  { semaine: 'S2', engagement: 72, reussite: 78 },
  { semaine: 'S3', engagement: 70, reussite: 76 },
  { semaine: 'S4', engagement: 75, reussite: 80 },
  { semaine: 'S5', engagement: 78, reussite: 82 },
  { semaine: 'S6', engagement: 82, reussite: 85 },
  { semaine: 'S7', engagement: 80, reussite: 84 },
  { semaine: 'S8', engagement: 85, reussite: 87 },
];

export default function Analytics({ onNavigate, onLogout }) {
  return (
    <div className="flex">
      <Sidebar currentPage="analytics" onNavigate={onNavigate} onLogout={onLogout} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-[#1E293B] mb-2">Analytics & Rapports</h1>
              <p className="text-[#64748B]">
                Analyse détaillée des performances et de l&apos;engagement des étudiants
              </p>
            </div>
            <div className="flex gap-3">
              <button className="flex items-center gap-2 px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition">
                <Download className="w-4 h-4" />
                Exporter rapport
              </button>
              <button className="flex items-center gap-2 px-4 py-2 bg-white border border-[#E2E8F0] rounded-lg text-[#334155] hover:bg-[#F8FAFC] transition">
                Générer benchmark
              </button>
            </div>
          </div>
        </div>

        {/* Cartes de statistiques globales */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#DBEAFE] p-3 rounded-lg">
                <Users className="w-6 h-6 text-[#2563EB]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Étudiants actifs</p>
            <p className="text-[#1E293B]">142 / 155</p>
            <p className="text-[#22C55E] mt-1">+5% vs semaine dernière</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#DCFCE7] p-3 rounded-lg">
                <TrendingUp className="w-6 h-6 text-[#22C55E]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Taux de réussite moyen</p>
            <p className="text-[#1E293B]">83%</p>
            <p className="text-[#22C55E] mt-1">+2% vs mois dernier</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#FED7AA] p-3 rounded-lg">
                <Activity className="w-6 h-6 text-[#F97316]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Engagement moyen</p>
            <p className="text-[#1E293B]">78%</p>
            <p className="text-[#F97316] mt-1">-1% vs semaine dernière</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#E0E7FF] p-3 rounded-lg">
                <Clock className="w-6 h-6 text-[#6366F1]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Temps moyen / semaine</p>
            <p className="text-[#1E293B]">8.5 heures</p>
            <p className="text-[#22C55E] mt-1">+0.5h vs semaine dernière</p>
          </div>
        </div>

        {/* Graphique : Taux de réussite par module */}
        <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
          <h3 className="text-[#1E293B] mb-4">Taux de réussite par module</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={moduleSuccessData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="module" stroke="#64748B" />
              <YAxis stroke="#64748B" />
              <Tooltip />
              <Bar dataKey="taux" fill="#2563EB" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Deux graphiques côte à côte */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Graphique : Nombre d'étudiants par profil */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Répartition des profils StudentProfiler</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={profileDistributionData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" stroke="#64748B" />
                <YAxis dataKey="profil" type="category" stroke="#64748B" width={120} />
                <Tooltip />
                <Bar dataKey="count" radius={[0, 8, 8, 0]}>
                  {profileDistributionData.map((entry, index) => {
                    const colors = ['#22C55E', '#F97316', '#EF4444', '#2563EB'];
                    return <rect key={`cell-${index}`} fill={colors[index]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Graphique : Évolution engagement vs réussite */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Évolution engagement vs réussite</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={engagementTrendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="semaine" stroke="#64748B" />
                <YAxis stroke="#64748B" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="engagement" stroke="#F97316" strokeWidth={2} name="Engagement" />
                <Line type="monotone" dataKey="reussite" stroke="#2563EB" strokeWidth={2} name="Réussite" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Heatmap des connexions (version simplifiée avec barres) */}
        <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
          <h3 className="text-[#1E293B] mb-4">Fréquence des connexions par jour et heure</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={weeklyActivityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="jour" stroke="#64748B" />
              <YAxis stroke="#64748B" />
              <Tooltip />
              <Legend />
              <Bar dataKey="h8" stackId="a" fill="#DBEAFE" name="8h-10h" />
              <Bar dataKey="h10" stackId="a" fill="#93C5FD" name="10h-12h" />
              <Bar dataKey="h14" stackId="a" fill="#3B82F6" name="14h-16h" />
              <Bar dataKey="h16" stackId="a" fill="#2563EB" name="16h-18h" />
              <Bar dataKey="h18" stackId="a" fill="#1E40AF" name="18h-20h" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Section benchmark */}
        <div className="bg-gradient-to-r from-[#EFF6FF] to-[#DBEAFE] rounded-2xl p-6 border border-[#93C5FD]">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-[#1E293B] mb-2">Génération de benchmark anonymisé</h3>
              <p className="text-[#64748B] mb-4">
                Créez un rapport anonymisé de vos données analytiques pour une publication académique (SoftwareX)
              </p>
              <ul className="space-y-2 text-[#64748B] mb-4">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#2563EB]"></div>
                  Anonymisation complète des données étudiants
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#2563EB]"></div>
                  Statistiques agrégées et métriques de performance
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#2563EB]"></div>
                  Format compatible avec les standards de publication
                </li>
              </ul>
            </div>
            <button className="px-6 py-3 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition whitespace-nowrap">
              Générer le benchmark
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}


