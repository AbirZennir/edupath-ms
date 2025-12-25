import { useEffect, useState } from 'react';
import { Download, TrendingUp, Activity, Users, Clock } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Analytics({ onNavigate, onLogout, user }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {

    fetch('http://localhost:8888/ai/dashboard/analytics')
      .then((res) => {
        if (!res.ok) throw new Error('Erreur lors du chargement des données');
        return res.json();
      })
      .then((data) => {
        setData(data);
        setLoading(false);
        console.log(data);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Chargement des analyses...</div>;
  if (error) return <div>Erreur : {error}</div>;

   const { stats, profileDistribution, evolution, moduleSuccess } = data;

  return (
    <div className="flex">
      <Sidebar currentPage="analytics" onNavigate={onNavigate} onLogout={onLogout} user={user} />

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
            </div>
          </div>
        </div>

        {/* Global Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#DBEAFE] p-3 rounded-lg">
                <Users className="w-6 h-6 text-[#2563EB]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Étudiants actifs</p>
            <p className="text-[#1E293B]">{stats.totalStudents}</p>
            <p className="text-[#22C55E] mt-1">+5% vs semaine dernière</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#DCFCE7] p-3 rounded-lg">
                <TrendingUp className="w-6 h-6 text-[#22C55E]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Taux de réussite moyen</p>
            <p className="text-[#1E293B]">{stats.successRate}%</p>
            <p className="text-[#22C55E] mt-1">+2% vs mois dernier</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#FED7AA] p-3 rounded-lg">
                <Activity className="w-6 h-6 text-[#F97316]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Engagement moyen</p>
            <p className="text-[#1E293B]">{stats.engagementAvg}</p>
            <p className="text-[#F97316] mt-1">-1% vs semaine dernière</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-[#E0E7FF] p-3 rounded-lg">
                <Clock className="w-6 h-6 text-[#6366F1]" />
              </div>
            </div>
            <p className="text-[#94A3B8] mb-1">Ressources consultées</p>
            <p className="text-[#1E293B]">{stats.resourcesConsulted}</p>
            <p className="text-[#22C55E] mt-1">+120 vs semaine dernière</p>
          </div>
        </div>

         <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
          <h3 className="text-[#1E293B] mb-4">Taux de réussite par module</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={moduleSuccess}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="module" stroke="#64748B" />
              <YAxis stroke="#64748B" />
              <Tooltip />
              <Bar dataKey="taux" fill="#2563EB" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Profiles and Evolution Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Répartition des profils StudentProfiler</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={profileDistribution} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" stroke="#64748B" />
                <YAxis dataKey="name" type="category" stroke="#64748B" width={120} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                  {profileDistribution.map((entry, index) => {
                    const colors = ['#22C55E', '#F97316', '#EF4444', '#2563EB'];
                    return <rect key={`cell-${index}`} fill={entry.color || colors[index % colors.length]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Évolution engagement vs réussite</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={evolution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="week" stroke="#64748B" />
                <YAxis stroke="#64748B" />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="engagement" stroke="#F97316" strokeWidth={2} name="Engagement" />
                <Line type="monotone" dataKey="success" stroke="#2563EB" strokeWidth={2} name="Réussite" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </main>
    </div>
  );
}