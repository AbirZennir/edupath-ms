import { useState } from 'react';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import ForgotPasswordPage from './components/ForgotPasswordPage';
import Dashboard from './components/Dashboard';
import ClassesModules from './components/ClassesModules';
import ClassDetail from './components/ClassDetail';
import StudentsAtRisk from './components/StudentsAtRisk';
import Recommendations from './components/Recommendations';
import Analytics from './components/Analytics';
import Settings from './components/Settings';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [authPage, setAuthPage] = useState('login');
  const [selectedClass, setSelectedClass] = useState(null);
  const [authToken, setAuthToken] = useState(null);

  const handleLogin = (token) => {
    setIsLoggedIn(true);
    setAuthToken(token);
    localStorage.setItem('authToken', token);
    setCurrentPage('dashboard');
  };

  const handleSignup = (token) => {
    setIsLoggedIn(true);
    setAuthToken(token);
    localStorage.setItem('authToken', token);
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setAuthToken(null);
    localStorage.removeItem('authToken');
    setAuthPage('login');
    setCurrentPage('dashboard');
  };

  const handleNavigate = (page) => {
    setCurrentPage(page);
    if (page !== 'class-detail') {
      setSelectedClass(null);
    }
  };

  const handleSelectClass = (classId) => {
    setSelectedClass(classId);
    setCurrentPage('class-detail');
  };

  if (!isLoggedIn) {
    if (authPage === 'signup') {
      return (
        <SignupPage
          onSignup={handleSignup}
          onBackToLogin={() => setAuthPage('login')}
        />
      );
    }

    if (authPage === 'forgot-password') {
      return <ForgotPasswordPage onBackToLogin={() => setAuthPage('login')} />;
    }

    return (
      <LoginPage
        onLogin={handleLogin}
        onForgotPassword={() => setAuthPage('forgot-password')}
        onSignup={() => setAuthPage('signup')}
      />
    );
  }

  return (
    <div className="min-h-screen bg-[#F5F7FB]">
      {currentPage === 'dashboard' && <Dashboard onNavigate={handleNavigate} onLogout={handleLogout} />}
      {currentPage === 'classes' && <ClassesModules onNavigate={handleNavigate} onSelectClass={handleSelectClass} onLogout={handleLogout} />}
      {currentPage === 'class-detail' && <ClassDetail onNavigate={handleNavigate} classId={selectedClass} onLogout={handleLogout} />}
      {currentPage === 'at-risk' && <StudentsAtRisk onNavigate={handleNavigate} onLogout={handleLogout} />}
      {currentPage === 'recommendations' && <Recommendations onNavigate={handleNavigate} onLogout={handleLogout} />}
      {currentPage === 'analytics' && <Analytics onNavigate={handleNavigate} onLogout={handleLogout} />}
      {currentPage === 'settings' && <Settings onNavigate={handleNavigate} onLogout={handleLogout} />}
    </div>
  );
}

export default App;
