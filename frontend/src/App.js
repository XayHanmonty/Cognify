import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';
import PDFUpload from './components/PDFUpload/PDFUpload';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const handleFileUpload = (file) => {
    // Handle the uploaded file here
    console.log('Uploaded file:', file);
    // You can add your file processing logic here
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={
            <>
              <Dashboard />
              <PDFUpload onFileUpload={handleFileUpload} />
            </>
          } />
        </Routes>
      </div>
    </ThemeProvider>
  );
}

export default App;
