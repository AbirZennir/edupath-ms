import { useEffect, useState } from 'react';
import { AlertTriangle, Filter, Mail, ClipboardList } from 'lucide-react';
import Sidebar from './Sidebar';

export default function StudentsAtRisk({ onNavigate, onLogout, user }) {
  const [atRiskStudents, setAtRiskStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8082/at-risk-students')
      .then((res) => {
        if (!res.ok) throw new Error('Erreur lors du chargement des étudiants');
        return res.json();
      })
      .then((data) => {
        setAtRiskStudents(data);
        setLoading(false);
        console.log(data);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const eleveCount = atRiskStudents.filter(s => s.niveau === 'elevé').length;

  if (loading) return <div>Chargement...</div>;
  if (error) return <div>Erreur : {error}</div>;

  return (
    <div className="flex">
      <Sidebar currentPage="at-risk" onNavigate={onNavigate} onLogout={onLogout} user={user} />
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-6">Étudiants à risque</h1>
          {/* Bandeau d'alerte */}
          <div className="bg-[#FEE2E2] border-l-4 border-[#EF4444] p-6 rounded-lg mb-6">
            <div className="flex items-start gap-4">
              <div className="bg-[#EF4444] p-3 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-[#1E293B] mb-1">Attention prioritaire requise</h3>
                <p className="text-[#64748B]">
                  {eleveCount} étudiants nécessitent une intervention immédiate. 
                  {atRiskStudents.length - eleveCount} autres étudiants présentent un risque modéré.
                </p>
              </div>
            </div>
          </div>
          {/* Filtres */}
          <div className="flex gap-4 mb-6">
            <button className="flex items-center gap-2 px-6 py-3 bg-white border border-[#E2E8F0] rounded-lg text-[#334155] hover:bg-[#F8FAFC] transition">
              <Filter className="w-5 h-5" />
              Tous les niveaux
            </button>
            <button className="px-6 py-3 bg-[#FEE2E2] text-[#EF4444] rounded-lg hover:bg-[#FECACA] transition">
              Risque élevé ({eleveCount})
            </button>
            <button className="px-6 py-3 bg-white border border-[#E2E8F0] text-[#64748B] rounded-lg hover:bg-[#F8FAFC] transition">
              Risque moyen ({atRiskStudents.length - eleveCount})
            </button>
          </div>
        </div>
        {/* Liste des étudiants */}
        <div className="grid gap-4">
          {atRiskStudents.map((student, idx) => (
            <div
              key={student.id && student.id !== 0 ? student.id : `${idx}-${student.nom}`}
              className={`bg-white rounded-2xl p-6 shadow-sm border-l-4 ${
                student.niveau === 'elevé' ? 'border-[#EF4444]' : 'border-[#F97316]'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4 flex-1">
                  {/* Avatar */}
                  <div className="w-12 h-12 rounded-full bg-[#2563EB] flex items-center justify-center text-white">
                    {student.avatar}
                  </div>
                  {/* Infos étudiant */}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-[#1E293B]">{student.nom}</h3>
                      <span
                        className={`px-3 py-1 rounded-full ${
                          student.niveau === 'elevé'
                            ? 'bg-[#FEE2E2] text-[#EF4444]'
                            : 'bg-[#FED7AA] text-[#F97316]'
                        }`}
                      >
                        Risque {student.niveau}
                      </span>
                    </div>
                    <div className="flex items-center gap-6 text-[#64748B]">
                      <span>{student.classe}</span>
                      <span>•</span>
                      <span>Modules : {student.modules}</span>
                      <span>•</span>
                      <span>Dernière connexion : Il y a une semaine</span>
                    </div>
                  </div>
                </div>
                {/* Probabilité d'échec */}
                <div className="text-right ml-6">
                  <p className="text-[#94A3B8] mb-2">Probabilité d&apos;échec</p>
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-[#E2E8F0] rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${
                          student.risque > 70 ? 'bg-[#EF4444]' : 'bg-[#F97316]'
                        }`}
                        style={{ width: `${student.risque}%` }}
                      ></div>
                    </div>
                    <span className="text-[#1E293B] min-w-[3rem]">{student.risque}%</span>
                  </div>
                </div>
              </div>
              {/* Actions */}
              <div className="flex gap-3 mt-4 pt-4 border-t border-[#F1F5F9]">
                <button className="flex items-center gap-2 px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition">
                  <ClipboardList className="w-4 h-4" />
                  Plan d&apos;action
                </button>
                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition">
                  <Mail className="w-4 h-4" />
                  Contacter
                </button>
                <button className="px-4 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition">
                  Voir recommandations
                </button>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}