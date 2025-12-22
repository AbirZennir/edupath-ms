import { useState } from 'react';
import { ArrowLeft, Users, TrendingUp, Activity, Clock } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';


const studentTimelineData = [
  { semaine: 'S1', note: 12, connexions: 8 },
  { semaine: 'S2', note: 14, connexions: 10 },
  { semaine: 'S3', note: 11, connexions: 6 },
  { semaine: 'S4', note: 15, connexions: 12 },
  { semaine: 'S5', note: 16, connexions: 14 },
  { semaine: 'S6', note: 17, connexions: 15 },
];

const recommendations = [
  { id: 1, titre: 'Tutoriel : Les arbres binaires', type: 'Vidéo', difficulte: 'Moyen' },
  { id: 2, titre: 'Quiz de révision : Complexité algorithmique', type: 'Quiz', difficulte: 'Facile' },
  { id: 3, titre: 'Exercices pratiques : Tri par fusion', type: 'PDF', difficulte: 'Difficile' },
];

const profilColors = {
  'Assidu': 'bg-[#DCFCE7] text-[#22C55E]',
  'Procrastinateur': 'bg-[#FED7AA] text-[#F97316]',
  'En difficulté': 'bg-[#FEE2E2] text-[#EF4444]',
  'Très performant': 'bg-[#DBEAFE] text-[#2563EB]',
};

export default function ClassDetail({ onNavigate, classId, onLogout, user }) {
  const [activeTab, setActiveTab] = useState('general');
  const [selectedStudent, setSelectedStudent] = useState(null);

  const handleBack = () => {
    onNavigate('classes');
  };

  const selectedStudentData = students.find(s => s.id === selectedStudent);

  return (
    <div className="flex">
      <Sidebar currentPage="classes" onNavigate={onNavigate} onLogout={onLogout} user={user} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={handleBack}
            className="flex items-center gap-2 text-[#2563EB] hover:text-[#1E40AF] mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            Retour aux classes
          </button>
          
          <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
            <h1 className="text-[#1E293B] mb-4">L3 Informatique – Algorithmique</h1>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-[#94A3B8] mb-1">Enseignant</p>
                <p className="text-[#1E293B]">Marie Petit</p>
              </div>
              <div>
                <p className="text-[#94A3B8] mb-1">Semestre</p>
                <p className="text-[#1E293B]">Semestre 5</p>
              </div>
              <div>
                <p className="text-[#94A3B8] mb-1">Étudiants</p>
                <p className="text-[#1E293B]">45 inscrits</p>
              </div>
              <div>
                <p className="text-[#94A3B8] mb-1">Taux de réussite</p>
                <p className="text-[#1E293B]">85%</p>
              </div>
            </div>
          </div>

          {/* Onglets */}
          <div className="flex gap-4 border-b border-[#E2E8F0]">
            <button
              onClick={() => setActiveTab('general')}
              className={`pb-3 px-2 transition ${
                activeTab === 'general'
                  ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                  : 'text-[#64748B] hover:text-[#334155]'
              }`}
            >
              Vue générale
            </button>
            <button
              onClick={() => setActiveTab('students')}
              className={`pb-3 px-2 transition ${
                activeTab === 'students'
                  ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                  : 'text-[#64748B] hover:text-[#334155]'
              }`}
            >
              Étudiants
            </button>
            <button
              onClick={() => setActiveTab('recommendations')}
              className={`pb-3 px-2 transition ${
                activeTab === 'recommendations'
                  ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                  : 'text-[#64748B] hover:text-[#334155]'
              }`}
            >
              Recommandations
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`pb-3 px-2 transition ${
                activeTab === 'history'
                  ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                  : 'text-[#64748B] hover:text-[#334155]'
              }`}
            >
              Historique modèle
            </button>
          </div>
        </div>

        {/* Contenu des onglets */}
        {activeTab === 'general' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-[#DBEAFE] p-3 rounded-lg">
                  <Users className="w-6 h-6 text-[#2563EB]" />
                </div>
                <div>
                  <p className="text-[#94A3B8]">Étudiants actifs</p>
                  <p className="text-[#1E293B]">42 / 45</p>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-[#DCFCE7] p-3 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-[#22C55E]" />
                </div>
                <div>
                  <p className="text-[#94A3B8]">Progression moyenne</p>
                  <p className="text-[#1E293B]">78%</p>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-[#FED7AA] p-3 rounded-lg">
                  <Activity className="w-6 h-6 text-[#F97316]" />
                </div>
                <div>
                  <p className="text-[#94A3B8]">Engagement</p>
                  <p className="text-[#1E293B]">82%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'students' && (
          <div className="flex gap-6">
            <div className={selectedStudent ? 'w-2/3' : 'w-full'}>
              <div className="bg-white rounded-2xl shadow-sm overflow-hidden">
                <table className="w-full">
                  <thead className="bg-[#F8FAFC]">
                    <tr>
                      <th className="text-left py-4 px-6 text-[#64748B]">Nom étudiant</th>
                      <th className="text-left py-4 px-6 text-[#64748B]">Profil</th>
                      <th className="text-left py-4 px-6 text-[#64748B]">Réussite</th>
                      <th className="text-left py-4 px-6 text-[#64748B]">Engagement</th>
                      <th className="text-left py-4 px-6 text-[#64748B]">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {students.map((student) => (
                      <tr key={student.id} className="border-b border-[#F1F5F9] hover:bg-[#F8FAFC]">
                        <td className="py-4 px-6">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-[#2563EB] flex items-center justify-center text-white">
                              {student.avatar}
                            </div>
                            <span className="text-[#1E293B]">{student.nom}</span>
                          </div>
                        </td>
                        <td className="py-4 px-6">
                          <span className={`px-3 py-1 rounded-full ${profilColors[student.profil]}`}>
                            {student.profil}
                          </span>
                        </td>
                        <td className="py-4 px-6">
                          <div className="flex items-center gap-2">
                            <div className="w-20 bg-[#E2E8F0] rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  student.reussite < 50 ? 'bg-[#EF4444]' : student.reussite < 70 ? 'bg-[#F97316]' : 'bg-[#22C55E]'
                                }`}
                                style={{ width: `${student.reussite}%` }}
                              ></div>
                            </div>
                            <span className="text-[#1E293B]">{student.reussite}%</span>
                          </div>
                        </td>
                        <td className="py-4 px-6 text-[#1E293B]">{student.engagement}%</td>
                        <td className="py-4 px-6">
                          <button
                            onClick={() => setSelectedStudent(student.id)}
                            className="text-[#2563EB] hover:text-[#1E40AF]"
                          >
                            Voir fiche
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Fiche étudiant (panneau latéral) */}
            {selectedStudent && selectedStudentData && (
              <div className="w-1/3">
                <div className="bg-white rounded-2xl p-6 shadow-sm sticky top-8">
                  <div className="flex justify-between items-start mb-6">
                    <h3 className="text-[#1E293B]">Fiche étudiant</h3>
                    <button
                      onClick={() => setSelectedStudent(null)}
                      className="text-[#64748B] hover:text-[#1E293B]"
                    >
                      ✕
                    </button>
                  </div>

                  <div className="text-center mb-6">
                    <div className="w-20 h-20 rounded-full bg-[#2563EB] flex items-center justify-center text-white mx-auto mb-3">
                      {selectedStudentData.avatar}
                    </div>
                    <h4 className="text-[#1E293B] mb-1">{selectedStudentData.nom}</h4>
                    <span className={`inline-block px-3 py-1 rounded-full ${profilColors[selectedStudentData.profil]}`}>
                      {selectedStudentData.profil}
                    </span>
                  </div>

                  {/* Timeline */}
                  <div className="mb-6">
                    <h4 className="text-[#334155] mb-3">Évolution des notes et connexions</h4>
                    <ResponsiveContainer width="100%" height={150}>
                      <LineChart data={studentTimelineData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis dataKey="semaine" stroke="#64748B" style={{ fontSize: '12px' }} />
                        <YAxis stroke="#64748B" style={{ fontSize: '12px' }} />
                        <Tooltip />
                        <Line type="monotone" dataKey="note" stroke="#2563EB" strokeWidth={2} />
                        <Line type="monotone" dataKey="connexions" stroke="#22C55E" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Alertes */}
                  <div className="mb-6">
                    <h4 className="text-[#334155] mb-3">Alertes PathPredictor</h4>
                    <div className="space-y-2">
                      <div className="bg-[#FEE2E2] border-l-4 border-[#EF4444] p-3 rounded">
                        <p className="text-[#EF4444]">Risque d&apos;échec élevé (78%)</p>
                      </div>
                      <div className="bg-[#FED7AA] border-l-4 border-[#F97316] p-3 rounded">
                        <p className="text-[#F97316]">Absence depuis 3 jours</p>
                      </div>
                    </div>
                  </div>

                  {/* Recommandations */}
                  <div>
                    <h4 className="text-[#334155] mb-3">Recommandations</h4>
                    <div className="space-y-2">
                      {recommendations.slice(0, 2).map((reco) => (
                        <div key={reco.id} className="border border-[#E2E8F0] rounded-lg p-3">
                          <p className="text-[#1E293B] mb-1">{reco.titre}</p>
                          <div className="flex items-center gap-2">
                            <span className="text-[#94A3B8]">{reco.type}</span>
                            <span className="text-[#94A3B8]">•</span>
                            <span className="text-[#94A3B8]">{reco.difficulte}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'recommendations' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <h3 className="text-[#1E293B] mb-4">Recommandations collectives</h3>
              <div className="grid gap-4">
                <div className="border border-[#E2E8F0] rounded-lg p-4">
                  <h4 className="text-[#1E293B] mb-2">Revoir le chapitre 3 : Structures de données</h4>
                  <p className="text-[#64748B] mb-3">25% des étudiants ont des difficultés sur ce chapitre</p>
                  <button className="text-[#2563EB] hover:text-[#1E40AF]">
                    Programmer une séance de révision
                  </button>
                </div>
                <div className="border border-[#E2E8F0] rounded-lg p-4">
                  <h4 className="text-[#1E293B] mb-2">Proposer une séance de tutorat</h4>
                  <p className="text-[#64748B] mb-3">15 étudiants pourraient bénéficier d&apos;un accompagnement personnalisé</p>
                  <button className="text-[#2563EB] hover:text-[#1E40AF]">
                    Organiser le tutorat
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h3 className="text-[#1E293B] mb-4">Historique du modèle PathPredictor</h3>
            <div className="space-y-4">
              <div className="border-l-4 border-[#2563EB] pl-4">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-[#64748B]" />
                  <span className="text-[#64748B]">8 décembre 2025, 10:30</span>
                </div>
                <p className="text-[#1E293B]">Mise à jour du modèle - Précision : 92%</p>
              </div>
              <div className="border-l-4 border-[#22C55E] pl-4">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-[#64748B]" />
                  <span className="text-[#64748B]">1 décembre 2025, 09:15</span>
                </div>
                <p className="text-[#1E293B]">Entraînement réussi - Précision : 89%</p>
              </div>
              <div className="border-l-4 border-[#F97316] pl-4">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-[#64748B]" />
                  <span className="text-[#64748B]">24 novembre 2025, 14:45</span>
                </div>
                <p className="text-[#1E293B]">Ajout de nouvelles features - Test en cours</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}


