import { useState, useEffect } from 'react';
import { ArrowLeft, Users, TrendingUp, Activity, Clock } from 'lucide-react';
import Sidebar from './Sidebar';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../api/client';

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

export default function ClassDetail({ onNavigate, classId, onLogout, user, token }) {
  const [activeTab, setActiveTab] = useState('general');
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStudents() {
      if (!classId || !token) return;
      try {
        setLoading(true);
        const data = await api.getCourseStudents(classId, token);
        setStudents(data);
      } catch (err) {
        console.error("Failed to fetch students", err);
      } finally {
        setLoading(false);
      }
    }
    fetchStudents();
  }, [classId, token]);

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
            <h1 className="text-[#1E293B] mb-4">Module {classId}</h1>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-[#94A3B8] mb-1">Enseignant</p>
                <p className="text-[#1E293B]">Enseignant</p>
              </div>
              <div>
                <p className="text-[#94A3B8] mb-1">Semestre</p>
                <p className="text-[#1E293B]">{classId.split('_')[1]}</p>
              </div>
              <div>
                <p className="text-[#94A3B8] mb-1">Étudiants</p>
                <p className="text-[#1E293B]">{students.length} inscrits</p>
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
                className="pb-4 px-2 text-sm font-medium border-b-2 border-[#2563EB] text-[#2563EB]"
              
            >
              Étudiants
            </button>
          </div>
        </div>

       

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
                                className={`h-2 rounded-full ${student.reussite < 50 ? 'bg-[#EF4444]' : student.reussite < 70 ? 'bg-[#F97316]' : 'bg-[#22C55E]'
                                  }`}
                                style={{ width: `${student.reussite}%` }}
                              ></div>
                            </div>
                            <span className="text-[#1E293B]">{student.reussite}%</span>
                          </div>
                        </td>
                        <td className="py-4 px-6 text-[#1E293B]">{student.engagement}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            
          </div>
        

     

        
      </main>
    </div>
  );
}


