import axios from 'axios';
import { useCallback, useState } from 'react';
import { NextPageContext } from 'next';
import { getSession, signIn } from 'next-auth/react';
import { useRouter } from 'next/router';
import { FcGoogle } from 'react-icons/fc';

import Input from '@/components/Input';

export async function getServerSideProps(context: NextPageContext) {
  const session = await getSession(context);

  if (session) {
    return {
      redirect: {
        destination: '/',
        permanent: false,
      }
    }
  }

  return {
    props: {}
  }
}

const Auth = () => {
  const router = useRouter();

  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');

  const [variant, setVariant] = useState('login');

  const toggleVariant = useCallback(() => {
    setVariant((currentVariant) => currentVariant === 'login' ? 'register' : 'login');
  }, []);

  const login = useCallback(async () => {
    try {
      await signIn('credentials', {
        email,
        password,
        redirect: false,
        callbackUrl: '/'
      });

      router.push('/');
    } catch (error) {
      console.log(error);
    }
  }, [email, password, router]);

  const register = useCallback(async () => {
    try {
      await axios.post('/api/register', {
        email,
        name,
        password
      });

      login();
    } catch (error) {
        console.log(error);
    }
  }, [email, name, password, login]);

  return (
    <div className="relative">
      <div className= "w-full h-full">
        <div className="flex justify-center">
          <div className="bg-black bg-opacity-70 px-16 py-16 self-center mt-2 lg:w-2/5 lg:max-w-md rounded-md w-full">
            <h2 className="text-white text-3xl mb-8 font-semibold flex justify-center">
              {variant === 'login' ? 'Sign in' : 'Register'}
            </h2>
            <div className="flex flex-col gap-4">
              {variant === 'register' && (
                <Input
                  id="name"
                  type="text"
                  label="Username"
                  value={name}
                  onChange={(e: any) => setName(e.target.value)} 
                />
              )}
              <Input
                id="email"
                type="email"
                label="Email"
                value={email}
                onChange={(e: any) => setEmail(e.target.value)} 
              />
              <Input
                type="password" 
                id="password" 
                label="Password" 
                value={password}
                onChange={(e: any) => setPassword(e.target.value)} 
              />
            </div>
            <button onClick={variant === 'login' ? login : register} className="bg-green-600 py-3 text-white rounded-md w-full mt-10 hover:bg-green-700 transition">
              {variant === 'login' ? 'Login' : 'Sign up'}
            </button>
            <div className="flex flex-row items-center gap-4 mt-8 justify-center">
              <div onClick={() => signIn('google', { callbackUrl: '/' })} className="w-10 h-10 bg-white rounded-full flex items-center justify-center cursor-pointer hover:opacity-80 transition">
                <FcGoogle size={32} />
              </div>
            </div>
            <p className="text-neutral-500 mt-12">
              {variant === 'login' ? 'First time using Netflix?' : 'Already have an account?'}
              <span onClick={toggleVariant} className="text-white ml-1 hover:underline cursor-pointer">
                {variant === 'login' ? 'Create an account' : 'Login'}
              </span>
              .
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Auth;
