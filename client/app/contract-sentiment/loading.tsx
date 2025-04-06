export default function Loading() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex justify-between items-center mb-8">
        <div className="text-2xl font-bold text-blue-600">ContractSentinel</div>
        <div className="text-sm text-gray-500">Loading analysis...</div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="bg-white rounded-lg shadow-md p-5 text-center"
          >
            <div className="animate-pulse">
              <div className="h-8 bg-gray-200 rounded w-16 mx-auto mb-2"></div>
              <div className="h-4 bg-gray-200 rounded w-24 mx-auto"></div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded mb-4"></div>
          <div className="flex justify-center gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-4 bg-gray-200 rounded w-24"></div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="py-3 px-4 text-left">
                    <div className="h-4 bg-gray-200 rounded w-24"></div>
                  </th>
                  <th className="py-3 px-4 text-left">
                    <div className="h-4 bg-gray-200 rounded w-20"></div>
                  </th>
                  <th className="py-3 px-4 text-left">
                    <div className="h-4 bg-gray-200 rounded w-24"></div>
                  </th>
                  <th className="py-3 px-4 text-left">
                    <div className="h-4 bg-gray-200 rounded w-20"></div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {[1, 2, 3, 4, 5].map((i) => (
                  <tr
                    key={i}
                    className={i < 5 ? "border-b border-gray-100" : ""}
                  >
                    <td className="py-3 px-4">
                      <div className="h-4 bg-gray-200 rounded w-32"></div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="h-6 bg-gray-200 rounded w-24"></div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="h-2 bg-gray-200 rounded w-full"></div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="h-6 bg-gray-200 rounded w-16"></div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
