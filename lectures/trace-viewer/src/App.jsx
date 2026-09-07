import { Routes, Route, BrowserRouter } from 'react-router-dom';
import './App.css'
import TraceViewer from './TraceViewer';

function App() {
  return (
    <BrowserRouter basename={window.location.pathname.replace(/\/index\.html$/, '')}>
      <Routes>
        <Route path="/" element={<TraceViewer />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;