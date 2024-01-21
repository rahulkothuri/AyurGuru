import Image from "next/image";
import Logo from "@/public/logo.png";
const Navbar = () => {
  return (
    <div className="navbar bg-base-100 drop-shadow-md">
      <div className="navbar-start">
        <div className="dropdown">
          <div tabIndex={0} role="button" className="btn btn-ghost lg:hidden">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h8m-8 6h16"
              />
            </svg>
          </div>
          <ul
            tabIndex={0}
            className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow bg-base-100 rounded-box w-52"
          >
            <li>
              <details>
                <summary>Info</summary>
                <ul className="p-2">
                  <li>
                    <a href="https://ayur-guru-six.vercel.app/info/prakriti">
                      Prakriti
                    </a>
                  </li>
                  <li>
                    <a href="https://ayur-guru-six.vercel.app/info/vata">
                      VATA
                    </a>
                  </li>
                  <li>
                    <a href="https://ayur-guru-six.vercel.app/info/pitta">
                      PITTA
                    </a>
                  </li>
                  <li>
                    <a href="https://ayur-guru-six.vercel.app/info/kapha">
                      KAPHA
                    </a>
                  </li>
                </ul>
              </details>
            </li>
            <li>
              <a href="https://ayur-guru-six.vercel.app/about-us">About Us</a>
            </li>
            <li>
              <a href="https://ayur-guru-six.vercel.app/contact-us">
                Contact Us
              </a>
            </li>
          </ul>
        </div>
        <div className="flex items-center">
          <Image src={Logo} alt="Ayur Guru Logo" width={80} />
          <a
            className="btn btn-ghost text-xl"
            href="https://ayur-guru-six.vercel.app/"
          >
            Ayur Guru
          </a>
        </div>
      </div>
      <div className="navbar-center hidden lg:flex">
        <ul className="menu menu-horizontal px-1">
          <li>
            <details>
              <summary>Info</summary>
              <ul className="p-2">
                <li>
                  <a href="https://ayur-guru-six.vercel.app/info/prakriti">
                    Prakriti
                  </a>
                </li>
                <li>
                  <a href="https://ayur-guru-six.vercel.app/info/vata">
                    VATA</a>
                </li>
                <li>
                  <a href="https://ayur-guru-six.vercel.app/info/pitta">
                    PITTA
                  </a>
                </li>
                <li>
                  <a href="https://ayur-guru-six.vercel.app/info/kapha">
                    KAPHA
                  </a>
                </li>
              </ul>
            </details>
          </li>
          <li>
            <a href="https://ayur-guru-six.vercel.app/about-us">About Us</a>
          </li>
          <li>
            <a href="https://ayur-guru-six.vercel.app/contact-us">Contact Us</a>
          </li>
        </ul>
      </div>
      <div className="navbar-end">
        <a
          className="btn btn-outline btn-primary"
          href="https://console.dialogflow.com/api-client/demo/embedded/87240b21-7222-4d83-b612-6d47a56f73b1"
        >
          Ayur Bot
        </a>
      </div>
    </div>
  );
};
export default Navbar;
