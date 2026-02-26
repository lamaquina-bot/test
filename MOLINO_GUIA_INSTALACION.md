# GUÍA DE INSTALACIÓN - PROYECTO MOLINO
## Nieto Digital - Mascota Virtual Terapéutica

**Versión:** 1.0
**Fecha:** 2026-02-26
**Estado:** 78% Completado (7/9 fases)

---

## 📋 REQUISITOS PREVIOS

### **Software necesario:**
- Node.js 18+ (https://nodejs.org)
- npm 9+ (viene con Node.js)
- Git (https://git-scm.com)
- VS Code (recomendado)

### **Cuentas necesarias:**
- GitHub (para clonar repositorio)
- OpenAI API Key (para chat IA)
- Cuenta de hosting (Vercel, Railway, o similar)

---

## 🚀 PASO 1: CLONAR REPOSITORIO

```bash
# Crear directorio de proyectos
mkdir -p ~/proyectos
cd ~/proyectos

# Clonar repositorio (cuando esté disponible)
git clone https://github.com/lamaquina-bot/nieto-digital.git
cd nieto-digital
```

---

## 📦 PASO 2: INSTALAR DEPENDENCIAS

### **Backend:**

```bash
# Ir al directorio backend
cd backend

# Instalar dependencias
npm install

# Dependencias principales que se instalarán:
# - express (servidor web)
# - mongoose (base de datos MongoDB)
# - openai (API de ChatGPT)
# - cors (permisos CORS)
# - dotenv (variables de entorno)
# - typescript
# - jest (testing)
```

### **Frontend:**

```bash
# Ir al directorio frontend
cd ../frontend

# Instalar dependencias
npm install

# Dependencias principales que se instalarán:
# - react (framework UI)
# - react-native (móvil)
# - expo (framework React Native)
# - @react-navigation (navegación)
# - axios (peticiones HTTP)
# - typescript
```

---

## ⚙️ PASO 3: CONFIGURACIÓN

### **3.1 Variables de entorno Backend**

Crear archivo `.env` en `backend/`:

```bash
# Backend/.env

# Puerto del servidor
PORT=3001

# URL de MongoDB (local o Atlas)
MONGODB_URI=mongodb://localhost:27017/nieto-digital
# O si usas MongoDB Atlas:
# MONGODB_URI=mongodb+srv://usuario:password@cluster.mongodb.net/nieto-digital

# API Key de OpenAI
OPENAI_API_KEY=sk-tu-api-key-aqui

# JWT Secret (generar uno aleatorio)
JWT_SECRET=tu-jwt-secret-super-seguro-aqui

# Entorno
NODE_ENV=development
```

### **3.2 Variables de entorno Frontend**

Crear archivo `.env` en `frontend/`:

```bash
# Frontend/.env

# URL del backend
API_URL=http://localhost:3001

# Configuración de Expo
EXPO_PUBLIC_API_URL=http://localhost:3001
```

---

## 🗄️ PASO 4: BASE DE DATOS

### **Opción A: MongoDB Local**

```bash
# Instalar MongoDB (Ubuntu/Debian)
sudo apt-get install mongodb

# Iniciar servicio
sudo systemctl start mongodb

# Verificar que está corriendo
sudo systemctl status mongodb
```

### **Opción B: MongoDB Atlas (Recomendado)**

1. Ir a https://www.mongodb.com/cloud/atlas
2. Crear cuenta gratuita
3. Crear cluster gratuito
4. Crear usuario de base de datos
5. Obtener URI de conexión
6. Agregar URI a `.env`

---

## 🏃 PASO 5: EJECUTAR APLICACIÓN

### **5.1 Iniciar Backend**

```bash
# Desde directorio backend
cd backend

# Modo desarrollo
npm run dev

# Verás algo como:
# Server running on port 3001
# MongoDB connected successfully
```

### **5.2 Iniciar Frontend**

```bash
# Desde directorio frontend
cd frontend

# Iniciar con Expo
npx expo start

# Se abrirá Metro Bundler
# Escanear QR con Expo Go (Android) o cámara (iOS)
```

---

## 📱 PASO 6: PROBAR EN DISPOSITIVO

### **6.1 Instalar Expo Go**

- **Android:** Play Store → "Expo Go"
- **iOS:** App Store → "Expo Go"

### **6.2 Conectar**

1. Asegurar que móvil y computadora estén en misma red WiFi
2. Escanear QR que aparece en terminal
3. La app se cargará en el dispositivo

---

## 🧪 PASO 7: EJECUTAR TESTS

### **Backend:**

```bash
cd backend

# Ejecutar todos los tests
npm test

# Ejecutar con cobertura
npm run test:coverage
```

### **Frontend:**

```bash
cd frontend

# Ejecutar tests
npm test
```

---

## 🏗️ PASO 8: COMPILAR PARA PRODUCCIÓN

### **8.1 Backend**

```bash
cd backend

# Compilar TypeScript
npm run build

# Los archivos compilados estarán en /dist
```

### **8.2 Frontend (Android)**

```bash
cd frontend

# Generar APK
npx expo build:android

# O usando EAS (recomendado)
npx eas build --platform android
```

### **8.3 Frontend (iOS)**

```bash
cd frontend

# Generar IPA
npx expo build:ios

# O usando EAS
npx eas build --platform ios
```

---

## 🚀 PASO 9: DESPLEGAR

### **9.1 Backend (Railway)**

```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Inicializar proyecto
railway init

# Desplegar
railway up
```

### **9.2 Frontend (Expo)**

```bash
# Publicar en Expo
npx expo publish

# O usar EAS
npx eas update
```

---

## 📂 ESTRUCTURA DEL PROYECTO

```
nieto-digital/
├── backend/
│   ├── src/
│   │   ├── index.ts          # Punto de entrada
│   │   ├── routes/           # Rutas API
│   │   ├── models/           # Modelos MongoDB
│   │   ├── services/         # Lógica de negocio
│   │   └── middleware/       # Middleware Express
│   ├── tests/                # Tests
│   ├── package.json
│   └── tsconfig.json
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Componente principal
│   │   ├── screens/          # Pantallas
│   │   ├── components/       # Componentes reutilizables
│   │   ├── hooks/            # Custom hooks
│   │   ├── context/          # Context API
│   │   └── integration/      # API Client
│   ├── tests/                # Tests
│   ├── app.json              # Config Expo
│   └── package.json
│
└── README.md
```

---

## 🔧 SOLUCIÓN DE PROBLEMAS

### **Error: "Cannot connect to MongoDB"**

```bash
# Verificar que MongoDB esté corriendo
sudo systemctl status mongodb

# Si no está corriendo
sudo systemctl start mongodb

# Verificar URI en .env
echo $MONGODB_URI
```

### **Error: "OpenAI API error"**

```bash
# Verificar API Key
echo $OPENAI_API_KEY

# Verificar que tenga créditos
# Ir a: https://platform.openai.com/account/usage
```

### **Error: "Expo no encuentra el backend"**

```bash
# Verificar que backend esté corriendo
curl http://localhost:3001/health

# Verificar API_URL en frontend/.env
cat frontend/.env

# Si usas dispositivo físico, usar IP local:
# API_URL=http://192.168.1.XXX:3001
```

### **Error: "Metro Bundler error"**

```bash
# Limpiar cache
npx expo start --clear

# Reinstalar dependencias
rm -rf node_modules
npm install
```

---

## 📊 ESTADO ACTUAL DEL PROYECTO

| Fase | Estado | Archivos |
|------|--------|----------|
| 1. Discovery | ✅ Completado | - |
| 2. Requirements | ✅ Completado | - |
| 3. Architecture | ✅ Completado | - |
| 4. UX/UI | ✅ Completado | - |
| 5. Backend | ✅ Completado | 5 archivos |
| 6. Frontend | ✅ Completado | 9 archivos |
| 7. Integration | ✅ Completado | 2 archivos |
| 8. DevOps | ⏳ Pendiente | - |
| 9. Security | ⏳ Pendiente | - |

**Total:** 16 archivos de código
**Tests:** 21 tests E2E
**Líneas código:** ~5,000+

---

## 🎯 PRÓXIMOS PASOS

1. **Completar DevOps:**
   - Configurar Docker
   - CI/CD con GitHub Actions
   - Deploy automatizado

2. **Completar Security:**
   - Auditoría de seguridad
   - Implementar HTTPS
   - Rate limiting

3. **Testing E2E:**
   - Tests en dispositivo real
   - Tests de usabilidad con adultos mayores

---

## 📞 SOPORTE

Si encuentras problemas:

1. Revisar logs del backend: `backend/logs/`
2. Revisar consola del navegador/dispositivo
3. Consultar documentación de Expo: https://docs.expo.dev
4. Consultar documentación de React Native: https://reactnative.dev

---

**Generado:** 2026-02-26
**Por:** Ines ☕✅
**Proyecto:** MOLINO - Nieto Digital
