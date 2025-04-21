import LoginForm from "@/components/auth/login-form";
import Header from "@/components/layout/header";
import Footer from "@/components/layout/footer";

export default function LoginPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1 py-12">
        <div className="container max-w-screen-xl mx-auto px-4">
          <LoginForm />
        </div>
      </main>
      <Footer />
    </div>
  );
}
