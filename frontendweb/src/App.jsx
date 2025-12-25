import { useState, useEffect } from 'react';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import ForgotPasswordPage from './components/ForgotPasswordPage';
import Dashboard from './components/Dashboard';
import ClassesModules from './components/ClassesModules';
import ClassDetail from './components/ClassDetail';
import StudentsAtRisk from './components/StudentsAtRisk';
import Analytics from './components/Analytics';
import Settings from './components/Settings';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [authPage, setAuthPage] = useState('login');
  const [selectedClass, setSelectedClass] = useState(null);
  const [authToken, setAuthToken] = useState(null);
  const [user, setUser] = useState(null);

  // Check for existing token on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('authToken');
    const storedUser = localStorage.getItem('user');
    if (storedToken) {
      setAuthToken(storedToken);
      setIsLoggedIn(true);
      if (storedUser) {
        setUser(JSON.parse(storedUser));
      }
    }
  }, []);

  const handleLogin = (token, userData) => {
    setIsLoggedIn(true);
    setAuthToken(token);
    setUser(userData);
    localStorage.setItem('authToken', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setCurrentPage('dashboard');
  };

  const handleSignup = (token, userData) => {
    setIsLoggedIn(true);
    setAuthToken(token);
    setUser(userData);
    localStorage.setItem('authToken', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setCurrentPage('dashboard');
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setAuthToken(null);
    setUser(null);
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
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
      {currentPage === 'dashboard' && <Dashboard onNavigate={handleNavigate} onLogout={handleLogout} user={user} />}
      {currentPage === 'classes' && <ClassesModules onNavigate={handleNavigate} onSelectClass={handleSelectClass} onLogout={handleLogout} user={user} token={authToken} />}
      {currentPage === 'class-detail' && <ClassDetail onNavigate={handleNavigate} classId={selectedClass} onLogout={handleLogout} user={user} token={authToken} />}
      {currentPage === 'at-risk' && <StudentsAtRisk onNavigate={handleNavigate} onLogout={handleLogout} user={user} />}
      {currentPage === 'analytics' && <Analytics onNavigate={handleNavigate} onLogout={handleLogout} user={user} />}
      {currentPage === 'settings' && <Settings onNavigate={handleNavigate} onLogout={handleLogout} user={user} />}
    </div>
  );
}

export default App;
