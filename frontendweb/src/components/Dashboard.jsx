import { useState, useEffect } from 'react';
import { RefreshCw, Download, TrendingUp, Users, Activity, BookOpen, AlertTriangle } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// API Configuration
const API_BASE_URL = 'http://localhost:8082/api';

export default function Dashboard({ onNavigate, onLogout, user }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch dashboard data from API
  const fetchDashboardData = async () => {
    try {
      setRefreshing(true);
      const response = await fetch(`${API_BASE_URL}/dashboard/analytics`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setDashboardData(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Initial data load
  useEffect(() => {
    fetchDashboardData();
  }, []);

  // Handle manual refresh
  const handleRefresh = () => {
    fetchDashboardData();
  };

  // Handle PDF export
  const handleExportPDF = () => {
    // TODO: Implement PDF export functionality
    alert('Export PDF functionality - To be implemented');
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex">
        <Sidebar currentPage="dashboard" onNavigate={onNavigate} onLogout={onLogout} user={user} />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-[#2563EB] mx-auto mb-4"></div>
            <p className="text-[#64748B]">Chargement des données...</p>
          </div>
        </main>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex">
        <Sidebar currentPage="dashboard" onNavigate={onNavigate} onLogout={onLogout} user={user} />
        <main className="flex-1 p-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-red-800 font-semibold mb-2">Erreur de chargement</h3>
            <p className="text-red-600 mb-4">{error}</p>
            <button 
              onClick={handleRefresh}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              Réessayer
            </button>
          </div>
        </main>
      </div>
    );
  }

  // Extract data from API response
  const stats = dashboardData?.stats || {};
  const atRiskStudents = dashboardData?.atRiskStudents || [];
  const profileData = dashboardData?.profileDistribution || [];
  const evolutionData = dashboardData?.evolution?.map(item => ({
    semaine: item.week,
    reussite: item.success,
    engagement: item.engagement
  })) || [];

  return (
    <div className="flex">
      <Sidebar currentPage="dashboard" onNavigate={onNavigate} onLogout={onLogout} user={user} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        
    

        {/* Cartes de statistiques */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#FEE2E2] p-3 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-[#EF4444]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Étudiants à risque</p>
            <p className="text-[#1E293B] text-3xl font-bold">{stats.studentsAtRisk || 0}</p>
            <p className="text-xs text-[#64748B] mt-1">sur {stats.totalStudents || 0} étudiants</p>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#DBEAFE] p-3 rounded-lg">
                <TrendingUp className="w-6 h-6 text-[#2563EB]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Taux de réussite global</p>
            <p className="text-[#1E293B] text-3xl font-bold">{stats.successRate || 0}%</p>
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
                <div 
                  className="bg-[#22C55E] h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${stats.engagementAvg || 0}%` }}
                ></div>
              </div>
              <p className="text-[#1E293B] text-2xl font-bold mt-1">{stats.engagementAvg || 0}%</p>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-[#FED7AA] p-3 rounded-lg">
                <BookOpen className="w-6 h-6 text-[#F97316]" />
              </div>
            </div>
            <p className="text-[#64748B] mb-1">Ressources consultées</p>
            <p className="text-[#1E293B] text-3xl font-bold">{stats.resourcesConsulted?.toLocaleString() || 0}</p>
          </div>
        </div>

        {/* Graphiques */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Graphique de courbes */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] text-xl font-semibold mb-4">Évolution dans le temps</h3>
            {evolutionData.length > 0 ? (
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
            ) : (
              <div className="h-[300px] flex items-center justify-center text-[#64748B]">
                Aucune donnée d&apos;évolution disponible
              </div>
            )}
          </div>

          {/* Graphique en camembert */}
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] text-xl font-semibold mb-4">Répartition des profils étudiants</h3>
            {profileData.length > 0 ? (
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
            ) : (
              <div className="h-[300px] flex items-center justify-center text-[#64748B]">
                Aucune donnée de profil disponible
              </div>
            )}
          </div>
        </div>

        {/* Tableau étudiants à risque */}
        <div className="bg-white rounded-2xl p-6 shadow-sm">
          <h3 className="text-[#1E293B] text-xl font-semibold mb-4">Top 10 étudiants à risque</h3>
          {atRiskStudents.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[#E2E8F0]">
                    <th className="text-left py-3 px-4 text-[#64748B] font-medium">Nom</th>
                    <th className="text-left py-3 px-4 text-[#64748B] font-medium">Classe</th>
                    <th className="text-left py-3 px-4 text-[#64748B] font-medium">Probabilité d&apos;échec</th>
                    <th className="text-left py-3 px-4 text-[#64748B] font-medium">Dernière connexion</th>
                    <th className="text-left py-3 px-4 text-[#64748B] font-medium">Statut</th>
                  </tr>
                </thead>
                <tbody>
                  {atRiskStudents.map((student) => (
                    <tr key={student.idStudent} className="border-b border-[#F1F5F9] hover:bg-[#F8FAFC] transition">
                      <td className="py-3 px-4 text-[#1E293B] font-medium">{student.name}</td>
                      <td className="py-3 px-4 text-[#64748B]">{student.className}</td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-24 bg-[#E2E8F0] rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-500 ${
                                student.riskScore > 70 ? 'bg-[#EF4444]' : 'bg-[#F97316]'
                              }`}
                              style={{ width: `${student.riskScore}%` }}
                            ></div>
                          </div>
                          <span className="text-[#1E293B] font-semibold">{student.riskScore}%</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-[#64748B]">{student.lastConnection}</td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium ${
                            student.status === 'critical'
                              ? 'bg-[#FEE2E2] text-[#EF4444]'
                              : 'bg-[#FED7AA] text-[#F97316]'
                          }`}
                        >
                          {student.status === 'critical' ? 'Critique' : 'Attention'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-8 text-center text-[#64748B]">
              Aucun étudiant à risque identifié
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
