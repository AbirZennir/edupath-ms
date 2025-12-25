import { useEffect, useState } from 'react';
import { AlertTriangle, Filter, Mail, ClipboardList, X, PlayCircle, FileText, Activity, Sparkles } from 'lucide-react';
import Sidebar from './Sidebar';
import { api } from '../api/client';

export default function StudentsAtRisk({ onNavigate, onLogout, user }) {
  const [atRiskStudents, setAtRiskStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // RecoBuilder State
  const [showRecoModal, setShowRecoModal] = useState(false);
  const [recoData, setRecoData] = useState(null);
  const [loadingReco, setLoadingReco] = useState(false);
  const [selectedStudentForReco, setSelectedStudentForReco] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8888/ai/at-risk-students')
      .then((res) => {
        if (!res.ok) throw new Error('Erreur lors du chargement des étudiants');
        return res.json();
      })
      .then((data) => {
        setAtRiskStudents(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const [selectedRisk, setSelectedRisk] = useState('all');

  const eleveCount = atRiskStudents.filter(s => s.status === 'critical').length;
  const moyenCount = atRiskStudents.length - eleveCount;

  const filteredStudents = selectedRisk === 'all'
    ? atRiskStudents
    : selectedRisk === 'elevé'
      ? atRiskStudents.filter(student => student.status === 'critical')
      : atRiskStudents.filter(student => student.status !== 'critical');

  const handleFilterClick = (riskLevel) => {
    if (selectedRisk === riskLevel) {
      setSelectedRisk('all');
    } else {
      setSelectedRisk(riskLevel);
    }
  };

  const handleShowReco = async (student) => {
    setSelectedStudentForReco(student);
    setShowRecoModal(true);
    setLoadingReco(true);


    const score = student.riskScore / 100;

    try {

      const token = localStorage.getItem('authToken');
      const data = await api.getRecommendations({
        riskScore: score,
        studentId: student.idStudent,
        codeModule: "AAA",
        codePresentation: "2013J"
      }, token);
      setRecoData(data);
    } catch (e) {
      console.error("Failed to load recommendations", e);
    } finally {
      setLoadingReco(false);
    }
  };

  const closeRecoModal = () => {
    setShowRecoModal(false);
    setRecoData(null);
  }

  if (loading) return <div>Chargement...</div>;
  if (error) return <div>Erreur : {error}</div>;

  return (
    <div className="flex relative">
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
                  {moyenCount} autres étudiants présentent un risque modéré.
                </p>
              </div>
            </div>
          </div>
          {/* Filtres */}
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => handleFilterClick('elevé')}
              className={`px-6 py-3 rounded-lg transition ${selectedRisk === 'elevé'
                ? 'bg-[#EF4444] text-white ring-2 ring-offset-2 ring-[#EF4444]'
                : 'bg-[#FEE2E2] text-[#EF4444] hover:bg-[#FECACA]'
                }`}
            >
              Risque élevé ({eleveCount})
            </button>
            <button
              onClick={() => handleFilterClick('moyen')}
              className={`px-6 py-3 rounded-lg transition ${selectedRisk === 'moyen'
                ? 'bg-[#F97316] text-white ring-2 ring-offset-2 ring-[#F97316]'
                : 'bg-white border border-[#E2E8F0] text-[#64748B] hover:bg-[#F8FAFC]'
                }`}
            >
              Risque moyen ({moyenCount})
            </button>
          </div>
        </div>
        {/* Liste des étudiants */}
        <div className="grid gap-4">
          {filteredStudents.map((student, idx) => (
            <div
              key={student.idStudent || idx}
              className={`bg-white rounded-2xl p-6 shadow-sm border-l-4 ${student.status === 'critical' ? 'border-[#EF4444]' : 'border-[#F97316]'
                }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4 flex-1">
                  {/* Avatar */}
                  <div className="w-12 h-12 rounded-full bg-[#2563EB] flex items-center justify-center text-white">
                    {student.avatar || "A"}
                  </div>
                  {/* Infos étudiant */}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-[#1E293B]">{student.name}</h3>
                      <span
                        className={`px-3 py-1 rounded-full ${student.status === 'critical'
                          ? 'bg-[#FEE2E2] text-[#EF4444]'
                          : 'bg-[#FED7AA] text-[#F97316]'
                          }`}
                      >
                        Risque {student.status === 'critical' ? 'élevé' : 'moyen'}
                      </span>
                    </div>
                    <div className="flex items-center gap-6 text-[#64748B]">
                      <span>{student.className}</span>
                      <span>•</span>
                      <span>Modules : {student.className.split(' ')[0]}</span>
                      <span>•</span>
                      <span>Dernière connexion : {student.lastConnection}</span>
                    </div>
                  </div>
                </div>
                {/* Probabilité d'échec */}
                <div className="text-right ml-6">
                  <p className="text-[#94A3B8] mb-2">Probabilité d&apos;échec</p>
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-[#E2E8F0] rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${student.riskScore > 70 ? 'bg-[#EF4444]' : 'bg-[#F97316]'
                          }`}
                        style={{ width: `${student.riskScore}%` }}
                      ></div>
                    </div>
                    <span className="text-[#1E293B] min-w-[3rem]">{student.riskScore}%</span>
                  </div>
                </div>
              </div>
              {/* Actions */}
              <div className="flex gap-3 mt-4 pt-4 border-t border-[#F1F5F9]">
                <button className="flex items-center gap-2 px-4 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition">
                  <Mail className="w-4 h-4" />
                  Contacter
                </button>
                <button
                  onClick={() => handleShowReco(student)}
                  className="flex items-center gap-2 px-4 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition"
                >
                  <Sparkles className="w-4 h-4 text-[#F59E0B]" />
                  Voir recommandations
                </button>
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* Reco Builder Modal */}
      {showRecoModal && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-y-auto">
          <div className="bg-white w-full max-w-4xl rounded-2xl shadow-xl p-8 max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold text-[#1E293B]">RecoBuilder : Recommandations Ciblées</h2>
                <p className="text-[#64748B]">
                  Pour {selectedStudentForReco?.name} (Risque: {selectedStudentForReco?.riskScore}%)
                </p>
              </div>
              <button onClick={closeRecoModal} className="p-2 hover:bg-gray-100 rounded-full">
                <X className="w-6 h-6 text-[#64748B]" />
              </button>
            </div>

            {loadingReco ? (
              <div className="py-12 text-center text-[#64748B]">Génération des recommandations en cours...</div>
            ) : recoData ? (
              <div className="space-y-8">
                {/* Profile Analysis */}
                <div className={`p-4 rounded-lg border-l-4 ${recoData.riskLevel === 'Élevé' ? 'bg-[#FEE2E2] border-[#EF4444]' :
                  recoData.riskLevel === 'Modéré' ? 'bg-[#FED7AA] border-[#F97316]' : 'bg-[#DCFCE7] border-[#22C55E]'
                  }`}>
                  <h3 className="font-bold text-[#1E293B] mb-1">Analyse du Profil</h3>
                  <p className="text-[#334155]">{recoData.studentProfile}</p>
                </div>

                {/* Recommendation Categories */}
                <div className="grid md:grid-cols-3 gap-6">
                  {recoData.categories.map((cat, idx) => (
                    <div key={idx} className="bg-[#F8FAFC] rounded-xl p-4 border border-[#E2E8F0]">
                      <div className="flex items-center gap-2 mb-4">
                        {cat.icon === 'Video' && <PlayCircle className="w-5 h-5" style={{ color: cat.color }} />}
                        {cat.icon === 'Article' && <FileText className="w-5 h-5" style={{ color: cat.color }} />}
                        {cat.icon === 'Exercise' && <Activity className="w-5 h-5" style={{ color: cat.color }} />}
                        <h4 className="font-bold text-[#1E293B]" style={{ color: cat.color }}>{cat.category}</h4>
                      </div>
                      <div className="space-y-3">
                        {cat.items.map((item) => (
                          <div key={item.id} className="bg-white p-3 rounded-lg shadow-sm border border-[#E2E8F0]">
                            <div className="flex justify-between items-start mb-1">
                              <h5 className="text-sm font-semibold text-[#1E293B]">{item.title}</h5>
                              {item.priority === 1 && <span className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded">Prioritaire</span>}
                            </div>
                            <p className="text-xs text-[#64748B] mb-2 line-clamp-2">{item.description}</p>
                            <div className="flex items-center justify-between text-xs text-[#94A3B8]">
                              <span>{item.duration}</span>
                              <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Accéder</a>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-[#64748B]">Aucune recommandation disponible</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}