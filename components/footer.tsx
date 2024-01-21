const Footer = () => {
    return (
      <div className="flex flex-col pt-10">
        <div className="flex-grow"></div>
        <div className="w-full py-5 text-center">
          <p className="text-gray-500">
            Made with ❤️ by{" "}
            <a
              className="font-semibold text-gray-600 underline-offset-4 transition-colors hover:underline"
              href="https://github.com/rahulkothuri/AyurGuru"
              target="_blank"
              rel="noopener noreferrer"
            >
              Rahul & Satya
            </a>
          </p>
        </div>
      </div>
    );
  };
  
  export default Footer;